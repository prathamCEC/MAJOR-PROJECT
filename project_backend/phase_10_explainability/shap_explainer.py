"""
Multimodal SHAP Attribution Explainer for Clinical and Retinal Features.

Computes exact and permutation-based SHAP (SHapley Additive exPlanations) values
for tabular health variables and multimodal input representations.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from phase_6_clinical_transformer.schema import ClinicalSchema


class MultimodalSHAPExplainer:
    """
    Computes game-theoretic SHAP feature attributions for Clinical Variables and Retinal Modalities.
    """

    def __init__(
        self,
        background_clinical_df: Optional[pd.DataFrame] = None,
        background_samples: int = 20,
        random_seed: int = 42,
    ):
        self.schema = ClinicalSchema()
        self.background_samples = background_samples
        self.rng = np.random.RandomState(random_seed)

        # Baseline clinical background reference
        if background_clinical_df is not None and not background_clinical_df.empty:
            self.background_df = background_clinical_df.copy()
        else:
            self.background_df = self._generate_synthetic_background(background_samples)

    def _generate_synthetic_background(self, n_samples: int) -> pd.DataFrame:
        """Generate statistically grounded reference baseline for background expectations."""
        rows = []
        for i in range(n_samples):
            rows.append({
                "ID#": f"BG_{i}",
                "Old groups": "O_CD" if i % 2 == 0 else "Y_CD",
                "Gender": int(i % 2),
                "Education": float(12.0 + (i % 6)),
                "BMI": float(22.0 + (i % 10) * 1.2),
                "Obese": 1.0 if (22.0 + (i % 10) * 1.2) >= 30.0 else 0.0,
                "EtOH_ever": int(i % 2),
                "EtOH_current": int(0),
                "Smoking_ever": int(i % 2),
                "Smoking_current": int(0),
                "HTN": int(i % 2),
                "DM2": int(i % 3 == 0),
            })
        return pd.DataFrame(rows)

    def explain_clinical_features(
        self,
        patient_record: Dict[str, Any],
        predict_fn: Callable[[pd.DataFrame], Dict[str, np.ndarray]],
        n_permutations: int = 25,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute marginal contribution SHAP values for each clinical feature.

        Args:
            patient_record: Dict of single patient variables
            predict_fn: Function mapping DataFrame -> {'stroke_logits': np.ndarray, 'alzheimer_logits': np.ndarray}
            n_permutations: Number of Monte Carlo feature subset permutations

        Returns:
            Dict containing 'stroke' and 'alzheimer' feature attribution dictionaries:
            {
                'stroke': {
                    'feature_names': [...],
                    'shap_values': [...],
                    'base_value': float,
                    'patient_values': {...}
                },
                'alzheimer': { ... }
            }
        """
        patient_df = pd.DataFrame([patient_record])

        # 1. Base Expected Prediction over Background
        bg_preds = predict_fn(self.background_df)
        base_stroke = float(np.mean(bg_preds["stroke_logits"]))
        base_alz = float(np.mean(bg_preds["alzheimer_logits"]))

        # 2. Patient Actual Prediction
        patient_preds = predict_fn(patient_df)
        actual_stroke = float(patient_preds["stroke_logits"][0])
        actual_alz = float(patient_preds["alzheimer_logits"][0])

        feature_cols = [c for c in patient_record.keys() if c != "ID#"]
        n_features = len(feature_cols)

        stroke_shap = {f: 0.0 for f in feature_cols}
        alz_shap = {f: 0.0 for f in feature_cols}

        # 3. Kernel / Marginal Permutation Sampling
        bg_subset = self.background_df.sample(
            n=min(len(self.background_df), self.background_samples),
            replace=True,
            random_state=42,
        ).reset_index(drop=True)

        for _ in range(n_permutations):
            perm = self.rng.permutation(feature_cols)
            # Iterate through feature ordering
            current_df = bg_subset.copy()

            for feat in perm:
                # Before adding feature
                pred_before = predict_fn(current_df)

                # Add patient's value for this feature
                current_df[feat] = patient_record[feat]

                # After adding feature
                pred_after = predict_fn(current_df)

                # Marginal difference
                diff_st = np.mean(pred_after["stroke_logits"] - pred_before["stroke_logits"])
                diff_al = np.mean(pred_after["alzheimer_logits"] - pred_before["alzheimer_logits"])

                stroke_shap[feat] += float(diff_st) / n_permutations
                alz_shap[feat] += float(diff_al) / n_permutations

        # Format Structured SHAP Dictionaries
        def format_task_shap(shap_dict: Dict[str, float], base_val: float, act_val: float) -> Dict[str, Any]:
            names = list(shap_dict.keys())
            vals = [shap_dict[k] for k in names]
            return {
                "feature_names": names,
                "shap_values": vals,
                "base_value": base_val,
                "actual_value": act_val,
                "patient_values": {k: patient_record.get(k) for k in names},
                "summary": [
                    {
                        "feature": k,
                        "value": patient_record.get(k),
                        "shap_value": round(shap_dict[k], 5),
                        "direction": "INCREASES_RISK" if shap_dict[k] > 0 else "DECREASES_RISK",
                    }
                    for k in sorted(shap_dict.keys(), key=lambda x: abs(shap_dict[x]), reverse=True)
                ],
            }

        return {
            "stroke": format_task_shap(stroke_shap, base_stroke, actual_stroke),
            "alzheimer": format_task_shap(alz_shap, base_alz, actual_alz),
        }

    def explain_modality_importance(
        self,
        retinal_weights: Dict[str, float],
        gate_weights: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute high-level multimodal attribution percentages across retinal and clinical pathways.
        """
        if isinstance(gate_weights, torch.Tensor):
            gate_val = float(torch.mean(gate_weights).item())
        else:
            gate_val = float(gate_weights)

        # Gate determines retinal vs clinical split:
        # v_fused = gate * v_ret + (1 - gate) * v_clin
        ret_share = gate_val
        clin_share = 1.0 - gate_val

        # Sub-divide retinal share by DMRA reliability weights
        octa_share = ret_share * retinal_weights.get("octa", 1/3)
        octb_share = ret_share * retinal_weights.get("octb", 1/3)
        fundus_share = ret_share * retinal_weights.get("fundus", 1/3)

        total = octa_share + octb_share + fundus_share + clin_share
        return {
            "octa_attribution_percent": round((octa_share / total) * 100.0, 2),
            "octb_attribution_percent": round((octb_share / total) * 100.0, 2),
            "fundus_attribution_percent": round((fundus_share / total) * 100.0, 2),
            "clinical_attribution_percent": round((clin_share / total) * 100.0, 2),
        }
