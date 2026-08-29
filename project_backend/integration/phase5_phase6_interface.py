"""
Integration Interface: Phase 5 (Unified Retinal Representation) + Phase 6 (Clinical Representation) -> Prepared for Phase 7.

Validates tensor dimension compatibility between Retinal (URR) and Clinical (CR)
representations and formats unified representations for downstream Phase 7 cross-attention.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import torch

from phase_5_retinal_fusion.config import get_default_fusion_config
from phase_5_retinal_fusion.fusion_model import RetinalMultimodalFusionModel
from phase_5_retinal_fusion.feature_loader import Phase4FeatureExtractor
from phase_6_clinical_transformer.config import get_default_clinical_config
from phase_6_clinical_transformer.feature_loader import ClinicalFeatureExtractor


class MultimodalPatientRepresentationBridge:
    """
    Coordinates simultaneous extraction of Retinal Representation (Phase 5)
    and Clinical Representation (Phase 6) for unified patient profiling.
    """

    def __init__(
        self,
        fusion_model: Optional[RetinalMultimodalFusionModel] = None,
        clinical_extractor: Optional[ClinicalFeatureExtractor] = None,
        device: str = "cpu",
    ):
        self.device = device
        # 1. Retinal Branch (Phase 4 -> Phase 5)
        self.retinal_feature_extractor = Phase4FeatureExtractor(device=device, pretrained_backbone=False)
        self.fusion_model = fusion_model or RetinalMultimodalFusionModel(config=get_default_fusion_config())
        self.fusion_model.to(torch.device(device))
        self.fusion_model.eval()

        # 2. Clinical Branch (Phase 6)
        self.clinical_extractor = clinical_extractor or ClinicalFeatureExtractor(device=device)

    def extract_unified_patient_inputs(
        self,
        patient_id: str,
        retinal_scans: Dict[str, Union[str, Path]],
        clinical_record: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Extract both representations for a single patient.

        Returns:
            Dict with:
            - 'patient_id': str
            - 'retinal_representation' (URR): Tensor [1, 512]
            - 'clinical_representation' (CR): Tensor [1, 512]
            - 'modality_weights': Dict of retinal reliability weights
            - 'is_phase7_compatible': bool
        """
        # A. Retinal URR Extraction (Phase 5)
        feats, mask = self.retinal_feature_extractor.extract_multimodal_patient_features(retinal_scans, pool=False)
        with torch.no_grad():
            fused_retina = self.fusion_model(modality_features=feats, modality_mask=mask)
        urr = fused_retina["urr"]  # [1, 512]

        # B. Clinical CR Extraction (Phase 6)
        if self.clinical_extractor.model is None:
            import pandas as pd
            df_dummy = pd.DataFrame([clinical_record])
            self.clinical_extractor.fit_and_initialize(df_dummy)

        clin_res = self.clinical_extractor.extract_single_patient(clinical_record)
        cr = clin_res["clinical_representation"]  # [1, 512]

        # Check compatibility
        compatible = (urr.shape[-1] == 512 and cr.shape[-1] == 512)

        return {
            "patient_id": patient_id,
            "retinal_representation": urr,
            "clinical_representation": cr,
            "retinal_weights": {m: w.item() for m, w in fused_retina["modality_weights"].items()},
            "is_phase7_compatible": compatible,
        }
