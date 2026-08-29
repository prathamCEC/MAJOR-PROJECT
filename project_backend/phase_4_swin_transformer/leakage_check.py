"""
Data Leakage Audit Module for Phase 4 Swin Transformer.

Verifies complete isolation across training, validation, and test splits:
- File path overlap detection
- SHA-256 pixel/content hash collision detection
- Patient-level cross-contamination detection when patient IDs are available
"""

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union
import pandas as pd


@dataclass
class LeakageCheckResult:
    """Outcome of data leakage audit."""
    passed: bool
    overlap_paths: List[str] = field(default_factory=list)
    hash_collisions: List[Tuple[str, str, str]] = field(default_factory=list)  # (splitA, splitB, hash)
    patient_overlaps: List[Tuple[str, str, str]] = field(default_factory=list)  # (splitA, splitB, patient_id)
    warnings: List[str] = field(default_factory=list)

    def format_summary(self) -> str:
        status_str = "PASS" if self.passed else "FAIL"
        lines = [
            "============================================================",
            f"DATA LEAKAGE AUDIT RESULT: [{status_str}]",
            "============================================================",
            f"Path Overlaps Detected   : {len(self.overlap_paths)}",
            f"Hash Collisions Detected : {len(self.hash_collisions)}",
            f"Patient Overlaps Detected: {len(self.patient_overlaps)}",
        ]

        if self.overlap_paths:
            lines.append("\nOverlapping File Paths:")
            for p in self.overlap_paths[:5]:
                lines.append(f"  * {p}")

        if self.hash_collisions:
            lines.append("\nHash Collisions (Identical Content Across Splits):")
            for sp1, sp2, h in self.hash_collisions[:5]:
                lines.append(f"  * Split '{sp1}' vs '{sp2}' (Hash: {h[:12]}...)")

        if self.patient_overlaps:
            lines.append("\nPatient Overlaps (Same Patient Across Splits):")
            for sp1, sp2, pid in self.patient_overlaps[:5]:
                lines.append(f"  * Patient ID '{pid}' in both '{sp1}' and '{sp2}'")

        if self.warnings:
            lines.append("\nWarnings:")
            for w in self.warnings:
                lines.append(f"  [!] {w}")

        lines.append("============================================================\n")
        return "\n".join(lines)


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of a file's content."""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def check_splits_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> LeakageCheckResult:
    """
    Check for path, content, and patient overlap across train, val, and test splits.
    """
    splits = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }

    overlap_paths: List[str] = []
    hash_collisions: List[Tuple[str, str, str]] = []
    patient_overlaps: List[Tuple[str, str, str]] = []
    warnings: List[str] = []

    # 1. Path overlap check
    split_names = list(splits.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            s1_name, s2_name = split_names[i], split_names[j]
            df1, df2 = splits[s1_name], splits[s2_name]

            s1_paths = set(df1["image_path"].astype(str))
            s2_paths = set(df2["image_path"].astype(str))
            common_paths = s1_paths.intersection(s2_paths)
            if common_paths:
                for cp in common_paths:
                    overlap_paths.append(f"[{s1_name} vs {s2_name}] {cp}")

            # Patient overlap check
            if "patient_id" in df1.columns and "patient_id" in df2.columns:
                s1_patients = set(df1["patient_id"].dropna().astype(str))
                s2_patients = set(df2["patient_id"].dropna().astype(str))
                common_patients = s1_patients.intersection(s2_patients)
                for pid in common_patients:
                    patient_overlaps.append((s1_name, s2_name, pid))
            else:
                warnings.append(
                    f"Patient-level leakage prevention cannot be fully guaranteed between {s1_name} and {s2_name} "
                    "because patient identifiers are unavailable in dataset manifest."
                )

    # 2. Hash collision check for available files
    hashes_by_split: Dict[str, Dict[str, str]] = {}  # split -> {hash: path}
    for s_name, df in splits.items():
        hashes_by_split[s_name] = {}
        for _, row in df.iterrows():
            img_p = Path(str(row["image_path"]))
            if img_p.exists() and img_p.is_file():
                h = compute_file_hash(img_p)
                hashes_by_split[s_name][h] = str(img_p)

    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            s1_name, s2_name = split_names[i], split_names[j]
            h1 = set(hashes_by_split[s1_name].keys())
            h2 = set(hashes_by_split[s2_name].keys())
            common_h = h1.intersection(h2)
            for ch in common_h:
                hash_collisions.append((s1_name, s2_name, ch))

    passed = (len(overlap_paths) == 0 and len(hash_collisions) == 0 and len(patient_overlaps) == 0)

    return LeakageCheckResult(
        passed=passed,
        overlap_paths=overlap_paths,
        hash_collisions=hash_collisions,
        patient_overlaps=patient_overlaps,
        warnings=warnings,
    )
