"""
Checkpoint Management Module for Clinical FT-Transformer.

Handles atomic saving and restoration of model weights, schema, preprocessor state,
and experiment configurations.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import torch

from .config import ClinicalTransformerConfig
from .clinical_model import ClinicalFTTransformerModel
from .preprocessing import ClinicalPreprocessor


class ClinicalCheckpointManager:
    """
    Manages checkpoints for Phase 6 Clinical FT-Transformer models.
    """

    def __init__(self, checkpoint_dir: Union[str, Path]):
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_path = self.checkpoint_dir / "best_clinical_model.pth"
        self.last_model_path = self.checkpoint_dir / "last_clinical_model.pth"
        self.preprocessor_path = self.checkpoint_dir / "preprocessor.json"

    def save_checkpoint(
        self,
        model: ClinicalFTTransformerModel,
        preprocessor: ClinicalPreprocessor,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        is_best: bool = False,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save model and preprocessor state.
        """
        prep_dict = preprocessor.to_dict()
        preprocessor.save_json(self.preprocessor_path)

        path = self.best_model_path if is_best else self.last_model_path
        saved_path = model.save_checkpoint(
            output_path=path,
            optimizer=optimizer,
            preprocessor_dict=prep_dict,
            epoch=epoch,
            metadata=extra_metadata,
        )

        if is_best and self.last_model_path != self.best_model_path:
            # Also update last model
            import shutil
            shutil.copy(str(saved_path), str(self.last_model_path))

        return saved_path

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cpu",
    ) -> Tuple[ClinicalFTTransformerModel, ClinicalPreprocessor, Dict[str, Any]]:
        """
        Load model, reconstruct preprocessor, and return metadata.
        """
        model, ckpt = ClinicalFTTransformerModel.load_checkpoint(
            checkpoint_path=checkpoint_path,
            optimizer=optimizer,
            device=device,
        )

        prep_dict = ckpt.get("preprocessor_state", {})
        if prep_dict:
            preprocessor = ClinicalPreprocessor.from_dict(prep_dict)
        else:
            preprocessor = ClinicalPreprocessor(schema=model.schema)

        return model, preprocessor, ckpt
