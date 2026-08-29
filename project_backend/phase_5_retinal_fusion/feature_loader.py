"""
Phase 4 Feature Extractor & Integration Adapter.

Provides a clean API to extract representations from Phase 4 Swin Transformer models
(OCT-A, OCT-B, Fundus) and feed them directly into Phase 5 Retinal Fusion.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image
import torch
import torch.nn as nn

from phase_4_swin_transformer.enums import Modality
from phase_4_swin_transformer.models.swin_factory import create_swin_model
from phase_4_swin_transformer.checkpoint import CheckpointManager
from phase_4_swin_transformer.transforms import get_transforms


class Phase4FeatureExtractor:
    """
    Adapter connecting Phase 4 Swin backbones to Phase 5 fusion.
    """

    def __init__(
        self,
        checkpoints: Optional[Dict[str, Union[str, Path]]] = None,
        device: str = "cpu",
        pretrained_backbone: bool = False,
    ):
        self.device = torch.device(device)
        self.checkpoints = checkpoints or {}
        self.models: Dict[str, nn.Module] = {}
        self.transforms: Dict[str, Any] = {}

        # Initialize encoders for each modality
        for mod_str in ["octa", "octb", "fundus"]:
            mod_enum = Modality.from_str(mod_str)
            # Create Swin model instance from Phase 4
            model = create_swin_model(
                modality=mod_enum,
                num_classes=2,
                pretrained=pretrained_backbone,
                model_name="swin_tiny_patch4_window7_224",
            )

            # Load trained weights if checkpoint supplied
            if mod_str in self.checkpoints and Path(self.checkpoints[mod_str]).exists():
                CheckpointManager.load_checkpoint(
                    checkpoint_path=self.checkpoints[mod_str],
                    model=model,
                    device=self.device,
                )

            model.to(self.device)
            model.eval()
            self.models[mod_str] = model

            # Setup eval transform
            self.transforms[mod_str] = get_transforms(
                modality=mod_enum,
                is_training=False,
                image_size=224,
            )

    def extract_from_image_path(
        self,
        image_path: Union[str, Path],
        modality: str,
        pool: bool = False,
    ) -> torch.Tensor:
        """
        Extract features from a single retinal scan file.

        Args:
            image_path: Path to image.
            modality: 'octa', 'octb', or 'fundus'.
            pool: If True, returns [1, 768]. If False, returns tokens [1, 49, 768].

        Returns:
            Extracted feature tensor.
        """
        mod_key = modality.lower()
        if mod_key not in self.models:
            raise KeyError(f"Modality '{modality}' not loaded in Phase4FeatureExtractor.")

        path = Path(image_path).resolve()
        with Image.open(path) as img:
            if mod_key == "fundus":
                img_pil = img.convert("RGB")
            else:
                img_pil = img.convert("L")

        tensor_img = self.transforms[mod_key](img_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.models[mod_key].extract_features(tensor_img, pool=pool)
        return feat

    def extract_multimodal_patient_features(
        self,
        patient_scans: Dict[str, Union[str, Path]],
        pool: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Extract features for all available scans of a given patient and construct modality mask.

        Args:
            patient_scans: Dict mapping modality name ('octa', 'octb', 'fundus') to file paths.
            pool: Return pooled vectors or token sequences.

        Returns:
            Tuple of:
            - features_dict: Dict mapping available modality names to feature tensors [1, N, D]
            - modality_mask: Dict mapping modality names to binary availability tensors [1, 1]
        """
        features_dict = {}
        modality_mask = {}

        for mod in ["octa", "octb", "fundus"]:
            if mod in patient_scans and patient_scans[mod] and Path(patient_scans[mod]).exists():
                feat = self.extract_from_image_path(patient_scans[mod], modality=mod, pool=pool)
                features_dict[mod] = feat
                modality_mask[mod] = torch.tensor([[1.0]], device=self.device)
            else:
                modality_mask[mod] = torch.tensor([[0.0]], device=self.device)

        return features_dict, modality_mask
