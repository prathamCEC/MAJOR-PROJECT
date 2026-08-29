"""
Inference Module for Phase 4 Swin Transformer.

Supports single-image and folder batch prediction using trained Swin Transformer
checkpoints with non-clinical research output formatting.
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F

from .enums import Modality, PredictionOutput
from .config import get_modality_config
from .models.swin_factory import create_swin_model
from .checkpoint import CheckpointManager
from .transforms import get_transforms
from .utils import get_device


class SwinInferenceEngine:
    """
    Production inference engine for Swin Transformer Retinal Classifiers.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        modality: Optional[Union[str, Modality]] = None,
        device: str = "auto",
    ):
        self.checkpoint_path = Path(checkpoint_path).resolve()
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {self.checkpoint_path}")

        self.device = get_device(device)

        # 1. Load Checkpoint Metadata
        raw_ckpt = torch.load(str(self.checkpoint_path), map_location="cpu")
        mod_str = modality or raw_ckpt.get("modality", "octa")
        self.modality = Modality.from_str(mod_str)
        self.class_mapping: Dict[str, int] = raw_ckpt.get("class_mapping", {"normal": 0, "disease": 1})
        self.idx_to_class: Dict[int, str] = {v: k for k, v in self.class_mapping.items()}
        self.num_classes = len(self.class_mapping)

        # 2. Build Model
        self.cfg = get_modality_config(self.modality)
        self.model = create_swin_model(
            modality=self.modality,
            num_classes=self.num_classes,
            pretrained=False,
            model_name=self.cfg.model_name,
        )
        CheckpointManager.load_checkpoint(self.checkpoint_path, self.model, device=self.device)
        self.model.to(self.device)
        self.model.eval()

        # 3. Setup Transform
        self.transform = get_transforms(
            modality=self.modality,
            is_training=False,
            image_size=self.cfg.image_size,
        )

    def predict_image(self, image_path: Union[str, Path]) -> PredictionOutput:
        """
        Run inference on a single retinal image file.
        """
        path = Path(image_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Image not found at path: {path}")

        with Image.open(path) as img:
            if self.modality == Modality.FUNDUS:
                image = img.convert("RGB")
            else:
                image = img.convert("L")

        tensor_img = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor_img)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            pred_idx = int(torch.argmax(logits, dim=1).item())

        pred_class = self.idx_to_class.get(pred_idx, f"class_{pred_idx}")
        confidence = float(probs[pred_idx])

        prob_dict = {
            self.idx_to_class.get(i, f"class_{i}"): float(probs[i])
            for i in range(self.num_classes)
        }

        return PredictionOutput(
            image_name=path.name,
            image_path=str(path),
            modality=self.modality.value,
            predicted_class=pred_class,
            predicted_label=pred_idx,
            confidence=confidence,
            probabilities=prob_dict,
        )

    def predict_batch(
        self,
        image_dir: Union[str, Path],
        output_csv_path: Optional[Union[str, Path]] = None,
    ) -> List[PredictionOutput]:
        """
        Run batch inference on all supported images in a directory.
        """
        in_dir = Path(image_dir).resolve()
        if not in_dir.exists():
            raise FileNotFoundError(f"Directory not found: {in_dir}")

        valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".ppm"}
        image_files = [f for f in in_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_exts]
        image_files.sort(key=lambda p: p.name)

        results: List[PredictionOutput] = []
        for img_p in image_files:
            try:
                res = self.predict_image(img_p)
                results.append(res)
            except Exception as e:
                print(f"Error predicting image '{img_p.name}': {e}")

        # Save to CSV
        if output_csv_path:
            csv_path = Path(output_csv_path).resolve()
            csv_path.parent.mkdir(parents=True, exist_ok=True)

            rows = []
            for r in results:
                row = {
                    "image_name": r.image_name,
                    "image_path": r.image_path,
                    "modality": r.modality,
                    "predicted_class": r.predicted_class,
                    "confidence": round(r.confidence, 4),
                }
                for c_name, p_val in r.probabilities.items():
                    row[f"prob_{c_name}"] = round(p_val, 4)
                rows.append(row)

            df = pd.DataFrame(rows)
            df.to_csv(csv_path, index=False)

        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 — Swin Transformer Model Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pth")
    parser.add_argument("--modality", type=str, default=None, choices=["octa", "octb", "fundus"], help="Modality")
    parser.add_argument("--image", type=str, default=None, help="Path to single image")
    parser.add_argument("--input", type=str, default=None, help="Path to batch directory")
    parser.add_argument("--output-csv", type=str, default=None, help="Path to save batch predictions CSV")

    args = parser.parse_args()

    engine = SwinInferenceEngine(checkpoint_path=args.checkpoint, modality=args.modality)

    if args.image:
        result = engine.predict_image(args.image)
        print("\n============================================================")
        print("PHASE 4 SWIN TRANSFORMER PREDICTION RESULT")
        print("============================================================")
        print(f"Image Name          : {result.image_name}")
        print(f"Modality            : {result.modality.upper()}")
        print(f"Predicted Class     : {result.predicted_class}")
        print(f"Model Confidence    : {result.confidence * 100:.2f}%")
        print("Class Probabilities :")
        for c_name, prob in sorted(result.probabilities.items()):
            print(f"  - {c_name:<18}: {prob * 100:6.2f}%")
        print(f"\nDisclaimer: {result.disclaimer}")
        print("============================================================\n")

    elif args.input:
        out_csv = args.output_csv or (Path(args.input) / "predictions.csv")
        results = engine.predict_batch(args.input, output_csv_path=out_csv)
        print(f"\nBatch inference completed for {len(results)} images. Saved results to: {out_csv}")


if __name__ == "__main__":
    main()
