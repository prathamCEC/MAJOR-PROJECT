"""
Batch Quality Assessment Processor for Retinal Images.

Processes entire folders of preprocessed images across OCT-A, OCT-B, and Fundus:
- Per-image fault isolation
- Summary statistics tracking (Total, Accepted, Warning, Rejected, Failed)
- Output reporting to logs/phase3_quality_results.csv and JSON
- Execution timing and progress tracking
"""

import csv
from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Dict, List, Optional, Union

from .config import (
    SUPPORTED_MODALITIES,
    SUPPORTED_IMAGE_EXTENSIONS,
    get_modality_quality_config,
    get_project_backend_root,
    get_default_processed_input_dir,
    get_default_phase3_log_dir,
)
from .pipeline import QualityAssessmentPipeline, AssessmentResult


@dataclass
class ModalityBatchStats:
    """Statistics for a single modality during batch quality assessment."""
    total: int = 0
    accepted: int = 0
    warning: int = 0
    rejected: int = 0
    failed: int = 0


@dataclass
class Phase3BatchSummary:
    """Overall Phase 3 batch execution summary."""
    modality_stats: Dict[str, ModalityBatchStats] = field(default_factory=dict)
    results: List[AssessmentResult] = field(default_factory=list)
    execution_time: float = 0.0
    csv_log_path: Optional[Path] = None
    json_log_path: Optional[Path] = None

    @property
    def total_images(self) -> int:
        return sum(s.total for s in self.modality_stats.values())

    @property
    def total_accepted(self) -> int:
        return sum(s.accepted for s in self.modality_stats.values())

    @property
    def total_warning(self) -> int:
        return sum(s.warning for s in self.modality_stats.values())

    @property
    def total_rejected(self) -> int:
        return sum(s.rejected for s in self.modality_stats.values())

    @property
    def total_failed(self) -> int:
        return sum(s.failed for s in self.modality_stats.values())


class Phase3BatchProcessor:
    """
    Batch processor executing Phase 3 Image Quality Assessment across datasets.
    """

    def __init__(
        self,
        input_dir: Optional[Union[str, Path]] = None,
        log_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the BatchProcessor.

        Args:
            input_dir: Directory containing processed images (under modality subdirectories).
            log_dir: Directory to store output CSV and JSON quality logs.
        """
        self.input_dir = (
            Path(input_dir).resolve()
            if input_dir
            else get_default_processed_input_dir()
        )
        self.log_dir = (
            Path(log_dir).resolve() if log_dir else get_default_phase3_log_dir()
        )
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.log_dir / "phase3_quality_results.csv"
        self.json_path = self.log_dir / "phase3_quality_results.json"

    def find_images(self, directory: Path) -> List[Path]:
        """Find all supported image files in a directory."""
        if not directory.exists() or not directory.is_dir():
            return []
        found: List[Path] = []
        for item in directory.iterdir():
            if item.is_file() and item.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                found.append(item)
        return sorted(found, key=lambda p: p.name.lower())

    def process_modality(
        self,
        modality: str,
        stats: ModalityBatchStats,
        all_results: List[AssessmentResult],
    ) -> None:
        """
        Assess all processed images for a specific modality.

        Args:
            modality: Modality identifier ('octa', 'octb', 'fundus').
            stats: ModalityBatchStats accumulator to update.
            all_results: Global result list to append to.
        """
        mod_dir = self.input_dir / modality
        if not mod_dir.exists():
            mod_dir.mkdir(parents=True, exist_ok=True)

        image_files = self.find_images(mod_dir)
        stats.total = len(image_files)

        if stats.total == 0:
            return

        pipeline = QualityAssessmentPipeline(modality=modality)

        for img_path in image_files:
            try:
                res = pipeline.assess_file(img_path)
                all_results.append(res)

                if res.decision == "ACCEPT":
                    stats.accepted += 1
                elif res.decision == "WARNING":
                    stats.warning += 1
                else:
                    stats.rejected += 1
            except Exception as exc:
                stats.failed += 1
                error_msg = f"{type(exc).__name__}: {str(exc)}"
                failed_res = AssessmentResult(
                    image_name=img_path.name,
                    modality=modality,
                    raw_metrics={},
                    scores={},
                    overall_score=0.0,
                    decision="REJECT",
                    is_approved_for_ai=False,
                    reason=f"Assessment error: {error_msg}",
                    failed_checks=["Assessment execution failure"],
                    error=error_msg,
                )
                all_results.append(failed_res)

    def write_reports(self, results: List[AssessmentResult]) -> None:
        """
        Write comprehensive evaluation reports to CSV and JSON.

        Args:
            results: List of AssessmentResult records.
        """
        # 1. Write CSV
        fieldnames = [
            "image_name",
            "modality",
            "blur_score",
            "brightness_score",
            "contrast_score",
            "noise_score",
            "clipping_score",
            "content_score",
            "color_score",
            "overall_score",
            "decision",
            "is_approved_for_ai",
            "reason",
            "error",
        ]

        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow({
                    "image_name": r.image_name,
                    "modality": r.modality,
                    "blur_score": r.scores.get("blur_score", ""),
                    "brightness_score": r.scores.get("brightness_score", ""),
                    "contrast_score": r.scores.get("contrast_score", ""),
                    "noise_score": r.scores.get("noise_score", ""),
                    "clipping_score": r.scores.get("clipping_score", ""),
                    "content_score": r.scores.get("content_score", ""),
                    "color_score": r.scores.get("color_score", "") if r.modality == "fundus" else "",
                    "overall_score": r.overall_score,
                    "decision": r.decision,
                    "is_approved_for_ai": r.is_approved_for_ai,
                    "reason": r.reason,
                    "error": r.error or "",
                })

        # 2. Write JSON
        serializable_results = [r.to_dict() for r in results]
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=2)

    def run(self, modality_filter: str = "all") -> Phase3BatchSummary:
        """
        Execute batch quality assessment for the selected modalities.

        Args:
            modality_filter: 'all', 'octa', 'octb', or 'fundus'.

        Returns:
            Phase3BatchSummary containing evaluation stats.
        """
        start_time = time.time()
        filter_clean = modality_filter.strip().lower()

        if filter_clean == "all":
            modalities_to_run = sorted(list(SUPPORTED_MODALITIES))
        elif filter_clean in SUPPORTED_MODALITIES:
            modalities_to_run = [filter_clean]
        else:
            raise ValueError(
                f"Invalid modality filter '{modality_filter}'. Must be 'all' or one of: {SUPPORTED_MODALITIES}"
            )

        summary = Phase3BatchSummary(
            csv_log_path=self.csv_path,
            json_log_path=self.json_path,
        )

        for mod in sorted(list(SUPPORTED_MODALITIES)):
            summary.modality_stats[mod] = ModalityBatchStats()

        for mod in modalities_to_run:
            self.process_modality(mod, summary.modality_stats[mod], summary.results)

        # Write reports
        self.write_reports(summary.results)

        summary.execution_time = round(time.time() - start_time, 2)
        return summary

    def format_summary(self, summary: Phase3BatchSummary) -> str:
        """
        Format the batch summary into a standardized human-readable report.
        """
        octa_st = summary.modality_stats.get("octa", ModalityBatchStats())
        octb_st = summary.modality_stats.get("octb", ModalityBatchStats())
        fundus_st = summary.modality_stats.get("fundus", ModalityBatchStats())

        report = (
            "============================================================\n"
            "PHASE 3 QUALITY ASSESSMENT SUMMARY\n"
            "============================================================\n\n"
            f"Total Images Assessed : {summary.total_images}\n\n"
            "OCT-A\n"
            f"Total    : {octa_st.total}\n"
            f"Accepted : {octa_st.accepted}\n"
            f"Warning  : {octa_st.warning}\n"
            f"Rejected : {octa_st.rejected}\n"
            f"Failed   : {octa_st.failed}\n\n"
            "OCT-B\n"
            f"Total    : {octb_st.total}\n"
            f"Accepted : {octb_st.accepted}\n"
            f"Warning  : {octb_st.warning}\n"
            f"Rejected : {octb_st.rejected}\n"
            f"Failed   : {octb_st.failed}\n\n"
            "FUNDUS\n"
            f"Total    : {fundus_st.total}\n"
            f"Accepted : {fundus_st.accepted}\n"
            f"Warning  : {fundus_st.warning}\n"
            f"Rejected : {fundus_st.rejected}\n"
            f"Failed   : {fundus_st.failed}\n\n"
            "Overall\n"
            f"Total    : {summary.total_images}\n"
            f"Accepted : {summary.total_accepted}\n"
            f"Warning  : {summary.total_warning}\n"
            f"Rejected : {summary.total_rejected}\n"
            f"Failed   : {summary.total_failed}\n\n"
            f"Execution Time : {summary.execution_time:.2f} seconds\n\n"
            "Reports Generated:\n"
            f"- {self.csv_path.name}\n"
            f"- {self.json_path.name}\n\n"
            "============================================================"
        )
        return report
