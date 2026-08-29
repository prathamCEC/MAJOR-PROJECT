"""
Batch Processing Engine for Phase 2 Retinal Image Preprocessing.

Supports high-throughput processing across OCT-A, OCT-B, and Fundus modalities with:
- Per-image fault isolation (a single failure never aborts the batch)
- Detailed summary statistics per modality and overall
- Rerun safety (skips previously processed files unless overwrite is requested)
- Error recording to logs/phase2_failed_images.txt
- Execution timing and progress tracking
"""

import argparse
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

from .config import (
    SUPPORTED_MODALITIES,
    get_modality_config,
    get_project_backend_root,
    get_default_raw_dir,
    get_default_processed_dir,
    get_default_log_file,
)
from .pipeline import PreprocessPipeline
from .utils import find_image_files, get_processed_filename


@dataclass
class ModalityStats:
    """Statistics for a single modality during batch processing."""
    total: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0


@dataclass
class BatchSummary:
    """Overall batch execution summary."""
    modality_stats: Dict[str, ModalityStats] = field(default_factory=dict)
    execution_time: float = 0.0
    output_dir: Path = field(default_factory=get_default_processed_dir)

    @property
    def total_images(self) -> int:
        return sum(s.total for s in self.modality_stats.values())

    @property
    def total_successful(self) -> int:
        return sum(s.successful for s in self.modality_stats.values())

    @property
    def total_failed(self) -> int:
        return sum(s.failed for s in self.modality_stats.values())

    @property
    def total_skipped(self) -> int:
        return sum(s.skipped for s in self.modality_stats.values())


class BatchProcessor:
    """
    Batch processor executing Phase 2 preprocessing across datasets.
    """

    def __init__(
        self,
        raw_dir: Optional[Union[str, Path]] = None,
        processed_dir: Optional[Union[str, Path]] = None,
        log_file: Optional[Union[str, Path]] = None,
        overwrite: bool = False,
    ):
        """
        Initialize the BatchProcessor.

        Args:
            raw_dir: Root directory containing raw images under modality subdirectories.
            processed_dir: Destination directory for processed images.
            log_file: Path to text file for logging failures.
            overwrite: If True, re-processes images even if output already exists.
        """
        self.raw_dir = Path(raw_dir).resolve() if raw_dir else get_default_raw_dir()
        self.processed_dir = (
            Path(processed_dir).resolve()
            if processed_dir
            else get_default_processed_dir()
        )
        self.log_file = (
            Path(log_file).resolve() if log_file else get_default_log_file()
        )
        self.overwrite = overwrite

        # Ensure directories exist
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def log_failure(self, modality: str, image_path: Path, error_message: str) -> None:
        """
        Append a failed image entry to the failure log file.

        Args:
            modality: Imaging modality identifier.
            image_path: Path to the image file that failed.
            error_message: Description of the error.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = (
            f"[{timestamp}] MODALITY={modality.upper()} | "
            f"FILE={image_path.name} | "
            f"PATH={image_path} | "
            f"ERROR={error_message}\n"
        )
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(entry)
        except Exception as e:
            print(f"Warning: Could not write to log file '{self.log_file}': {e}")

    def process_modality(
        self,
        modality: str,
        stats: ModalityStats,
    ) -> None:
        """
        Process all raw images found for a specific modality.

        Args:
            modality: Modality identifier ('octa', 'octb', 'fundus').
            stats: ModalityStats accumulator to update.
        """
        mod_raw_dir = self.raw_dir / modality
        mod_processed_dir = self.processed_dir / modality
        mod_processed_dir.mkdir(parents=True, exist_ok=True)

        if not mod_raw_dir.exists():
            mod_raw_dir.mkdir(parents=True, exist_ok=True)

        image_files = find_image_files(mod_raw_dir)
        stats.total = len(image_files)

        if stats.total == 0:
            return

        pipeline = PreprocessPipeline(modality=modality)

        for img_path in image_files:
            output_name = get_processed_filename(img_path, output_ext="png")
            out_path = mod_processed_dir / output_name

            # Rerun safety: check if output file already exists
            if out_path.exists() and not self.overwrite:
                stats.skipped += 1
                continue

            try:
                pipeline.process(input_path=img_path, output_path=out_path)
                stats.successful += 1
            except Exception as exc:
                stats.failed += 1
                error_msg = f"{type(exc).__name__}: {str(exc)}"
                self.log_failure(modality, img_path, error_msg)

    def run(self, modality_filter: str = "all") -> BatchSummary:
        """
        Execute batch processing for the selected modalities.

        Args:
            modality_filter: 'all', 'octa', 'octb', or 'fundus'.

        Returns:
            BatchSummary dataclass containing execution stats.
        """
        start_time = time.time()
        filter_clean = modality_filter.strip().lower()

        if filter_clean == "all":
            modalities_to_run = list(SUPPORTED_MODALITIES)
        elif filter_clean in SUPPORTED_MODALITIES:
            modalities_to_run = [filter_clean]
        else:
            raise ValueError(
                f"Invalid modality filter '{modality_filter}'. "
                f"Must be 'all' or one of: {SUPPORTED_MODALITIES}"
            )

        summary = BatchSummary(output_dir=self.processed_dir)

        # Initialize all modalities in summary
        for mod in SUPPORTED_MODALITIES:
            summary.modality_stats[mod] = ModalityStats()

        # Run selected modalities
        for mod in modalities_to_run:
            self.process_modality(mod, summary.modality_stats[mod])

        summary.execution_time = round(time.time() - start_time, 2)
        return summary

    def format_summary(self, summary: BatchSummary) -> str:
        """
        Format the batch summary into the standardized human-readable report.

        Args:
            summary: BatchSummary object.

        Returns:
            Formatted string.
        """
        octa_st = summary.modality_stats.get("octa", ModalityStats())
        octb_st = summary.modality_stats.get("octb", ModalityStats())
        fundus_st = summary.modality_stats.get("fundus", ModalityStats())

        # Relative or clean path for display
        try:
            rel_output = summary.output_dir.relative_to(get_project_backend_root())
        except ValueError:
            rel_output = summary.output_dir

        report = (
            "============================================================\n"
            "PHASE 2 PREPROCESSING SUMMARY\n"
            "============================================================\n\n"
            f"Total Images : {summary.total_images}\n\n"
            "OCT-A\n"
            f"Total      : {octa_st.total}\n"
            f"Successful : {octa_st.successful}\n"
            f"Failed     : {octa_st.failed}\n"
            f"Skipped    : {octa_st.skipped}\n\n"
            "OCT-B\n"
            f"Total      : {octb_st.total}\n"
            f"Successful : {octb_st.successful}\n"
            f"Failed     : {octb_st.failed}\n"
            f"Skipped    : {octb_st.skipped}\n\n"
            "FUNDUS\n"
            f"Total      : {fundus_st.total}\n"
            f"Successful : {fundus_st.successful}\n"
            f"Failed     : {fundus_st.failed}\n"
            f"Skipped    : {fundus_st.skipped}\n\n"
            "Overall\n"
            f"Total      : {summary.total_images}\n"
            f"Successful : {summary.total_successful}\n"
            f"Failed     : {summary.total_failed}\n"
            f"Skipped    : {summary.total_skipped}\n\n"
            f"Execution Time : {summary.execution_time:.2f} seconds\n\n"
            "Output:\n"
            f"{rel_output}\n\n"
            "============================================================"
        )
        return report


def main() -> None:
    """Command-line entry point for Phase 2 batch preprocessing."""
    parser = argparse.ArgumentParser(
        description="Phase 2 — Retinal Image Preprocessing Batch Processor"
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="all",
        choices=["all", "octa", "octb", "fundus"],
        help="Modality to process: 'all' (default), 'octa', 'octb', or 'fundus'",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=None,
        help="Path to raw dataset directory (default: datasets/raw/)",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=None,
        help="Path to processed output directory (default: datasets/processed/)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to failed images log file (default: logs/phase2_failed_images.txt)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Force reprocessing of existing processed images (default: False)",
    )

    args = parser.parse_args()

    processor = BatchProcessor(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        log_file=args.log_file,
        overwrite=args.overwrite,
    )

    summary = processor.run(modality_filter=args.modality)
    print(processor.format_summary(summary))


if __name__ == "__main__":
    main()
