"""
Patient Multimodal Fusion Pipeline Adapter.

Coordinates end-to-end extraction from Phase 5 (Retinal URR) and Phase 6 (Clinical CR)
into Phase 7 Unified Patient Representation (UPR) ready for downstream Phase 8.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import torch

from phase_5_retinal_fusion.config import get_default_fusion_config
from phase_5_retinal_fusion.fusion_model import RetinalMultimodalFusionModel
from phase_5_retinal_fusion.feature_loader import Phase4FeatureExtractor
from phase_6_clinical_transformer.config import get_default_clinical_config
from phase_6_clinical_transformer.feature_loader import ClinicalFeatureExtractor

from .config import RetinaClinicalConfig, get_default_retina_clinical_config
from .fusion_model import RetinaClinicalFusionModel


class PatientMultimodalPipeline:
    """
    End-to-End Orchestrator for Phase 4 -> Phase 5 -> Phase 6 -> Phase 7.
    """

    def __init__(
        self,
        config: Optional[RetinaClinicalConfig] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: str = "auto",
    ):
        self.config = config or get_default_retina_clinical_config()
        self.device = self.config.get_device() if device == "auto" else torch.device(device)

        # 1. Retinal Branch (Phase 4 -> Phase 5)
        self.retinal_feature_extractor = Phase4FeatureExtractor(device=str(self.device), pretrained_backbone=False)
        self.retinal_fusion_model = RetinalMultimodalFusionModel(config=get_default_fusion_config())
        self.retinal_fusion_model.to(self.device)
        self.retinal_fusion_model.eval()

        # 2. Clinical Branch (Phase 6)
        self.clinical_extractor = ClinicalFeatureExtractor(device=str(self.device))

        # 3. Phase 7 Fusion Model
        if checkpoint_path and Path(checkpoint_path).exists():
            self.model, self.ckpt_meta = RetinaClinicalFusionModel.load_checkpoint(
                checkpoint_path=checkpoint_path,
                device=str(self.device),
            )
        else:
            self.model = RetinaClinicalFusionModel(config=self.config)
            self.model.to(self.device)
        self.model.eval()

    def extract_patient_upr(
        self,
        patient_id: str,
        retinal_scans: Dict[str, Union[str, Path]],
        clinical_record: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute full pipeline for a single patient to generate their UPR.

        Args:
            patient_id: Patient unique identifier.
            retinal_scans: Dict of modality paths, e.g. {"octa": path, "octb": path, "fundus": path}.
            clinical_record: Dict of clinical attributes (BMI, Education, HTN, etc.).

        Returns:
            Dict containing:
            - 'patient_id': str
            - 'upr': Tensor [1, upr_dim] (e.g. [1, 512])
            - 'retinal_urr': Tensor [1, 512]
            - 'clinical_cr': Tensor [1, 512]
            - 'gate_weights': Tensor [1, 512]
            - 'retinal_weights': Dict of learned retinal modality weights
        """
        # Step A: Extract Retinal URR (Phase 4 -> Phase 5)
        feats, mask = self.retinal_feature_extractor.extract_multimodal_patient_features(retinal_scans, pool=False)
        with torch.no_grad():
            fused_retina = self.retinal_fusion_model(modality_features=feats, modality_mask=mask)
        urr = fused_retina["urr"].to(self.device)  # [1, 512]

        # Step B: Extract Clinical CR (Phase 6)
        if self.clinical_extractor.model is None:
            df_dummy = pd.DataFrame([clinical_record])
            self.clinical_extractor.fit_and_initialize(df_dummy)

        clin_res = self.clinical_extractor.extract_single_patient(clinical_record)
        cr = clin_res["clinical_representation"].to(self.device)  # [1, 512]

        # Step C: Phase 7 Cross-Attention & UPR Fusion
        with torch.no_grad():
            out_phase7 = self.model(
                retinal_representation=urr,
                clinical_representation=cr,
            )

        upr = out_phase7["upr"]  # [1, 512]

        return {
            "patient_id": patient_id,
            "upr": upr,
            "retinal_urr": urr,
            "clinical_cr": cr,
            "gate_weights": out_phase7["gate_weights"],
            "retinal_weights": {m: w.item() for m, w in fused_retina["modality_weights"].items()},
        }
