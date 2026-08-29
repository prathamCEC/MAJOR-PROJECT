"""
Clinical Feature Extractor and Inference Adapter.

Loads patient clinical data, applies fitted preprocessors, and runs the FT-Transformer
to extract standardized Clinical Representation (CR) vectors ready for Phase 7.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import torch

from .config import ClinicalTransformerConfig, get_default_clinical_config
from .schema import ClinicalSchema, get_default_retinal_clinical_schema
from .preprocessing import ClinicalPreprocessor
from .clinical_model import ClinicalFTTransformerModel
from .dataset import ClinicalTabularDataset, create_clinical_dataloader


class ClinicalFeatureExtractor:
    """
    High-level engine to extract Clinical Representations (CR) from tabular records.
    """

    def __init__(
        self,
        config: Optional[ClinicalTransformerConfig] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: str = "auto",
    ):
        self.config = config or get_default_clinical_config()
        self.device = self.config.get_device() if device == "auto" else torch.device(device)

        if checkpoint_path and Path(checkpoint_path).exists():
            self.model, self.preprocessor, self.ckpt_meta = ClinicalFTTransformerModel.load_checkpoint(
                checkpoint_path=checkpoint_path,
                device=str(self.device),
            )
            self.schema = self.model.schema
        else:
            self.schema = self.config.schema
            self.preprocessor = ClinicalPreprocessor(schema=self.schema)
            self.model = None

    def fit_and_initialize(self, df_train: pd.DataFrame) -> None:
        """
        Fit preprocessor on training DataFrame and initialize FT-Transformer model.
        """
        self.preprocessor.fit(df_train)
        cards = [
            self.preprocessor.state.category_cardinalities[col]
            for col in self.schema.all_categorical_like
        ]
        self.model = ClinicalFTTransformerModel(
            config=self.config,
            categorical_cardinalities=cards,
        )
        self.model.to(self.device)
        self.model.eval()

    def extract_representations(
        self,
        df: pd.DataFrame,
        batch_size: int = 16,
    ) -> Dict[str, Any]:
        """
        Extract Clinical Representation vectors for all records in df.

        Args:
            df: Clinical DataFrame.
            batch_size: Batch size.

        Returns:
            Dict containing:
            - 'clinical_representations': Tensor [N, clinical_representation_dim] (e.g. [N, 512])
            - 'patient_ids': List[str] corresponding patient identifiers
            - 'feature_tokens': Tensor [N, 1 + N_feat, embed_dim]
        """
        if self.model is None or not self.preprocessor.state.fitted:
            raise RuntimeError("Extractor must be initialized via fit_and_initialize() or loaded from a checkpoint.")

        num_mat, cat_mat, patient_ids = self.preprocessor.transform(df)
        dataset = ClinicalTabularDataset(
            numerical_matrix=num_mat,
            categorical_matrix=cat_mat,
            patient_ids=patient_ids,
        )
        loader = create_clinical_dataloader(dataset, batch_size=batch_size, shuffle=False)

        all_cr = []
        all_tokens = []
        all_pids = []

        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                x_num = batch["numerical_features"].to(self.device)
                x_cat = batch["categorical_features"].to(self.device)
                out = self.model(x_num, x_cat)

                all_cr.append(out["clinical_representation"].cpu())
                all_tokens.append(out["feature_tokens"].cpu())
                all_pids.extend(batch["patient_id"])

        return {
            "clinical_representations": torch.cat(all_cr, dim=0),
            "patient_ids": all_pids,
            "feature_tokens": torch.cat(all_tokens, dim=0),
        }

    def extract_single_patient(
        self,
        patient_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Extract Clinical Representation for a single patient record dictionary.
        """
        df = pd.DataFrame([patient_dict])
        res = self.extract_representations(df, batch_size=1)
        return {
            "patient_id": res["patient_ids"][0],
            "clinical_representation": res["clinical_representations"][0:1],
        }
