"""
Integration Pipeline: Phase 3 (Quality Approved) -> Phase 4 (Swin Transformer Dataset Preparation).

Locates approved images from Phase 3, associates clinical labels from metadata manifests
(e.g., 5_ASSOCIATED DATA.xlsx or folder labels), performs patient-level dataset partitioning,
and runs pre-training validation audits.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd

from phase_4_swin_transformer.enums import DiseaseTask, Modality
from phase_4_swin_transformer.config import get_approved_dataset_dir, get_splits_dir, get_project_root
from phase_4_swin_transformer.validation import DatasetValidator, DatasetReport
from phase_4_swin_transformer.split_dataset import create_dataset_splits
from phase_4_swin_transformer.leakage_check import check_splits_leakage, LeakageCheckResult


class Phase3ToPhase4Integrator:
    """
    Coordinates data flow from Phase 3 Approved Datasets into Phase 4 training manifests.
    """

    def __init__(
        self,
        approved_base_dir: Optional[Union[str, Path]] = None,
        clinical_metadata_path: Optional[Union[str, Path]] = None,
        splits_dir: Optional[Union[str, Path]] = None,
    ):
        self.approved_base_dir = (
            Path(approved_base_dir).resolve()
            if approved_base_dir
            else get_approved_dataset_dir()
        )
        self.splits_dir = (
            Path(splits_dir).resolve()
            if splits_dir
            else get_splits_dir()
        )
        self.splits_dir.mkdir(parents=True, exist_ok=True)

        root = get_project_root().parent  # MAJOR-PROJECT root
        default_meta = root / "5_ASSOCIATED DATA.xlsx"
        self.clinical_metadata_path = (
            Path(clinical_metadata_path).resolve()
            if clinical_metadata_path
            else (default_meta if default_meta.exists() else None)
        )

    def load_clinical_labels(self) -> Optional[pd.DataFrame]:
        """Load clinical metadata Excel/CSV if present."""
        if not self.clinical_metadata_path or not self.clinical_metadata_path.exists():
            return None

        try:
            if self.clinical_metadata_path.suffix.lower() in (".xlsx", ".xls"):
                df = pd.read_excel(self.clinical_metadata_path)
            else:
                df = pd.read_csv(self.clinical_metadata_path)
            return df
        except Exception as e:
            print(f"Warning: Could not read clinical metadata file: {e}")
            return None

    def build_modality_manifest(
        self,
        modality: Union[str, Modality],
        task: Union[str, DiseaseTask] = DiseaseTask.ALZHEIMERS,
    ) -> pd.DataFrame:
        """
        Scan Phase 3 approved directory and associate clinical labels.
        """
        mod_enum = Modality.from_str(modality) if isinstance(modality, str) else modality
        task_enum = DiseaseTask.from_str(task) if isinstance(task, str) else task
        mod_dir = self.approved_base_dir / mod_enum.value

        if not mod_dir.exists():
            mod_dir.mkdir(parents=True, exist_ok=True)

        valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        approved_files = [f for f in mod_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_exts]

        clinical_df = self.load_clinical_labels()
        rows: List[Dict] = []

        for f in approved_files:
            # Extract sample ID from filename (e.g. 'N10A_L_processed.png' or 'fundus_sample_1_processed.png')
            stem = f.stem.replace("_processed", "").replace("_sample", "")
            label_val = 0
            class_name = "normal"
            patient_id = stem

            if clinical_df is not None:
                # Attempt to match sample ID with clinical metadata table
                match = clinical_df[clinical_df["ID#"].astype(str).str.lower() == stem.lower()]
                if not match.empty:
                    if task_enum == DiseaseTask.ALZHEIMERS and "AD" in match.columns:
                        ad_val = int(match.iloc[0]["AD"])
                        label_val = ad_val
                        class_name = "alzheimers" if ad_val == 1 else "normal"
                    elif task_enum == DiseaseTask.STROKE and "HTN" in match.columns:
                        # Vascular risk proxy if stroke specific column absent
                        htn_val = int(match.iloc[0]["HTN"])
                        label_val = htn_val
                        class_name = "stroke_risk" if htn_val == 1 else "normal"
                else:
                    # Synthetic / demo assignment for testing pipeline if not explicitly in table
                    label_val = 0
                    class_name = "normal"

            rows.append({
                "image_path": str(f.resolve()),
                "modality": mod_enum.value,
                "label": label_val,
                "class_name": class_name,
                "patient_id": patient_id,
            })

        return pd.DataFrame(rows)

    def prepare_phase4_data(
        self,
        modality: Union[str, Modality],
        task: Union[str, DiseaseTask] = DiseaseTask.ALZHEIMERS,
        random_seed: int = 42,
    ) -> Tuple[DatasetReport, Optional[LeakageCheckResult]]:
        """
        Complete integration routine: Ingest approved data -> Partition -> Audit.
        """
        mod_enum = Modality.from_str(modality) if isinstance(modality, str) else modality
        task_enum = DiseaseTask.from_str(task) if isinstance(task, str) else task

        manifest_df = self.build_modality_manifest(mod_enum, task=task_enum)

        validator = DatasetValidator(task=task_enum)
        stats = validator.validate_csv_manifest(
            self.splits_dir / f"{mod_enum.value}_{task_enum.value}_manifest.csv"
            if (self.splits_dir / f"{mod_enum.value}_{task_enum.value}_manifest.csv").exists()
            else self.approved_base_dir / mod_enum.value,
            modality=mod_enum,
        )

        report = DatasetReport(
            modality_stats={mod_enum.value: stats},
            task=task_enum,
            is_ready_for_training=stats.has_verified_labels and stats.valid_images >= 6,
        )

        leakage_result = None
        if len(manifest_df) >= 6 and manifest_df["label"].nunique() >= 2:
            train_df, val_df, test_df = create_dataset_splits(
                manifest_df,
                modality=mod_enum,
                task=task_enum,
                random_seed=random_seed,
                output_dir=self.splits_dir,
            )
            leakage_result = check_splits_leakage(train_df, val_df, test_df)

        return report, leakage_result


def run_phase3_to_phase4(modality: str = "all", task: str = "alzheimers") -> None:
    """CLI execution function."""
    integrator = Phase3ToPhase4Integrator()
    mods = ["octa", "octb", "fundus"] if modality.lower() == "all" else [modality.lower()]

    print("\n============================================================")
    print("PHASE 3 -> PHASE 4 INTEGRATION PIPELINE")
    print("============================================================\n")

    for m in mods:
        print(f"> Processing Modality: {m.upper()}")
        manifest = integrator.build_modality_manifest(m, task=task)
        print(f"  Ingested Approved Images : {len(manifest)}")
        if not manifest.empty:
            print(f"  Class Breakdown          : {manifest['class_name'].value_counts().to_dict()}")
        
        # Save manifest
        manifest_p = integrator.splits_dir / f"{m}_{task}_manifest.csv"
        manifest.to_csv(manifest_p, index=False)
        print(f"  Exported Manifest to     : {manifest_p.name}\n")

    print("============================================================\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 -> Phase 4 Dataset Integration")
    parser.add_argument("--modality", type=str, default="all", choices=["all", "octa", "octb", "fundus"])
    parser.add_argument("--task", type=str, default="alzheimers", choices=["stroke", "alzheimers", "multi_disease"])

    args = parser.parse_args()
    run_phase3_to_phase4(modality=args.modality, task=args.task)


if __name__ == "__main__":
    main()
