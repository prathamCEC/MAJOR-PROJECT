"""
Dataset Validation Module for Phase 4 Swin Transformer.

Inspects dataset manifests and directories, verifies clinical label integrity,
checks for corrupted files, and produces standardized dataset audit reports.
"""

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
from PIL import Image

from .enums import DiseaseTask, Modality


@dataclass
class ModalityValidationStats:
    """Validation statistics for a single imaging modality."""
    total_images: int = 0
    class_distribution: Dict[str, int] = field(default_factory=dict)
    valid_images: int = 0
    missing_files: int = 0
    corrupted_files: int = 0
    has_patient_ids: bool = False
    patient_count: int = 0
    has_verified_labels: bool = False
    error_messages: List[str] = field(default_factory=list)


@dataclass
class DatasetReport:
    """Complete dataset audit report across all modalities."""
    modality_stats: Dict[str, ModalityValidationStats] = field(default_factory=dict)
    task: DiseaseTask = DiseaseTask.ALZHEIMERS
    is_ready_for_training: bool = False
    blocking_reasons: List[str] = field(default_factory=list)

    def format_report(self) -> str:
        """Render report as human-readable string."""
        lines = [
            "============================================================",
            "PHASE 4 DATASET VALIDATION REPORT",
            "============================================================",
            f"Target Task: {self.task.value.upper()}",
            "------------------------------------------------------------",
        ]

        for mod_name, stats in self.modality_stats.items():
            lines.append(f"MODALITY: {mod_name.upper()}")
            lines.append(f"  Total Images Ingested : {stats.total_images}")
            lines.append(f"  Valid / Loadable      : {stats.valid_images}")
            lines.append(f"  Missing / Broken Paths: {stats.missing_files}")
            lines.append(f"  Corrupted Files       : {stats.corrupted_files}")
            lines.append(f"  Verified Labels Exist : {stats.has_verified_labels}")
            lines.append(f"  Patient IDs Available : {stats.has_patient_ids} (Unique Patients: {stats.patient_count})")
            lines.append("  Class Distribution:")
            if stats.class_distribution:
                for c_name, count in sorted(stats.class_distribution.items()):
                    lines.append(f"    - {c_name:<18}: {count}")
            else:
                lines.append("    (No classes discovered)")

            if stats.error_messages:
                lines.append("  Errors/Warnings:")
                for err in stats.error_messages:
                    lines.append(f"    * {err}")
            lines.append("------------------------------------------------------------")

        lines.append(f"Ready for Supervised Training: {self.is_ready_for_training}")
        if self.blocking_reasons:
            lines.append("\nBLOCKING REASONS / ACTION REQUIRED:")
            for reason in self.blocking_reasons:
                lines.append(f"  [!] {reason}")
        lines.append("============================================================\n")

        return "\n".join(lines)


class DatasetValidator:
    """
    Validates dataset integrity, verifies labels, and audits data quality.
    """

    def __init__(self, task: Union[str, DiseaseTask] = DiseaseTask.ALZHEIMERS):
        self.task = DiseaseTask.from_str(task) if isinstance(task, str) else task

    def validate_csv_manifest(
        self,
        csv_path: Union[str, Path],
        modality: Union[str, Modality],
    ) -> ModalityValidationStats:
        """
        Validate a dataset manifest CSV.
        """
        path = Path(csv_path).resolve()
        mod_enum = Modality.from_str(modality) if isinstance(modality, str) else modality
        stats = ModalityValidationStats()

        if not path.exists():
            stats.error_messages.append(f"Manifest CSV not found: {path}")
            return stats

        try:
            df = pd.read_csv(path)
        except Exception as e:
            stats.error_messages.append(f"Failed to read CSV: {e}")
            return stats

        required_cols = {"image_path", "label"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            stats.error_messages.append(f"CSV missing required columns: {missing_cols}")
            return stats

        stats.total_images = len(df)
        if stats.total_images == 0:
            stats.error_messages.append("Dataset CSV is empty.")
            return stats

        # Verify class distribution
        class_col = "class_name" if "class_name" in df.columns else "label"
        counts = df[class_col].value_counts().to_dict()
        stats.class_distribution = {str(k): int(v) for k, v in counts.items()}

        # Verify verified labels
        if len(stats.class_distribution) >= 2:
            stats.has_verified_labels = True
        else:
            stats.error_messages.append(
                f"Dataset must contain at least 2 distinct verified classes for classification; found {len(stats.class_distribution)}."
            )

        # Check patient IDs
        if "patient_id" in df.columns and df["patient_id"].notna().any():
            stats.has_patient_ids = True
            stats.patient_count = df["patient_id"].nunique()

        # Check image files
        for _, row in df.iterrows():
            img_p = Path(str(row["image_path"]))
            if not img_p.is_absolute():
                img_p = (path.parent / img_p).resolve()

            if not img_p.exists():
                stats.missing_files += 1
            else:
                try:
                    with Image.open(img_p) as img:
                        img.verify()
                    stats.valid_images += 1
                except Exception:
                    stats.corrupted_files += 1

        return stats

    def validate_folder(
        self,
        folder_path: Union[str, Path],
        modality: Union[str, Modality],
    ) -> ModalityValidationStats:
        """
        Validate class subfolder directory.
        """
        path = Path(folder_path).resolve()
        stats = ModalityValidationStats()

        if not path.exists():
            stats.error_messages.append(f"Dataset directory not found: {path}")
            return stats

        valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".ppm"}
        class_dirs = [d for d in path.iterdir() if d.is_dir()]

        if not class_dirs:
            flat_files = [f for f in path.iterdir() if f.is_file() and f.suffix.lower() in valid_exts]
            stats.total_images = len(flat_files)
            stats.class_distribution = {"unlabeled": len(flat_files)}
            stats.has_verified_labels = False
            stats.error_messages.append(
                "Images are located in a flat folder without class labels. Verified labels are required for supervised training."
            )
            for f in flat_files:
                try:
                    with Image.open(f) as img:
                        img.verify()
                    stats.valid_images += 1
                except Exception:
                    stats.corrupted_files += 1
            return stats

        for c_dir in class_dirs:
            files = [f for f in c_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_exts]
            stats.class_distribution[c_dir.name] = len(files)
            stats.total_images += len(files)

            for f in files:
                try:
                    with Image.open(f) as img:
                        img.verify()
                    stats.valid_images += 1
                except Exception:
                    stats.corrupted_files += 1

        if len(stats.class_distribution) >= 2:
            stats.has_verified_labels = True
        else:
            stats.error_messages.append(
                f"Requires at least 2 distinct class folders; found {len(stats.class_distribution)}."
            )

        return stats


def validate_modality_dataset(
    data_source: Union[str, Path],
    modality: Union[str, Modality],
    task: Union[str, DiseaseTask] = DiseaseTask.ALZHEIMERS,
) -> ModalityValidationStats:
    """Convenience helper to validate a single modality dataset."""
    validator = DatasetValidator(task=task)
    path = Path(data_source).resolve()
    if path.is_file() and path.suffix.lower() == ".csv":
        return validator.validate_csv_manifest(path, modality)
    return validator.validate_folder(path, modality)
