"""
Integrated Phase 2 (Preprocessing) -> Phase 3 (Quality Assessment) Pipeline.

Orchestrates the continuous medical imaging AI preparation pipeline:
1. RAW IMAGE INGESTION
2. PHASE 2: Modality-Specific Standardization & Preprocessing
3. PHASE 3: Non-Destructive Technical Quality Assessment
4. DATASET ROUTING: Approved (Ready for Phase 4) vs Rejected
5. COMPREHENSIVE LOGGING & REPORTING
"""

import argparse
import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Dict, List, Optional, Union

# Phase 2 Real API
from phase_2_image_preprocessing.src.pipeline import PreprocessPipeline
from phase_2_image_preprocessing.src.utils import find_image_files, get_processed_filename
from phase_2_image_preprocessing.src.config import SUPPORTED_MODALITIES as PHASE2_MODALITIES

# Phase 3 Real API
from phase_3_image_quality_assessment.src.pipeline import QualityAssessmentPipeline, AssessmentResult
from phase_3_image_quality_assessment.src.config import SUPPORTED_MODALITIES as PHASE3_MODALITIES

from .config import (
    get_raw_dir,
    get_processed_dir,
    get_approved_dir,
    get_rejected_dir,
    get_integration_log_dir,
    get_project_root,
)


@dataclass
class IntegratedItemResult:
    """Detailed record for a single image processed through the integrated pipeline."""
    raw_filename: str
    processed_filename: str
    modality: str
    phase2_status: str  # "SUCCESS" or "FAILED"
    phase3_status: str  # "SUCCESS", "FAILED", or "SKIPPED"
    overall_quality_score: Optional[float]
    decision: str       # "ACCEPT", "WARNING", "REJECT", or "ERROR"
    is_approved_for_ai: bool
    routed_location: str
    reason: str
    error: Optional[str] = None


@dataclass
class IntegratedModalityStats:
    """Statistics per modality for integrated pipeline execution."""
    total_raw: int = 0
    phase2_success: int = 0
    phase2_failed: int = 0
    phase3_assessed: int = 0
    approved: int = 0
    rejected: int = 0
    errors: int = 0


@dataclass
class IntegratedPipelineSummary:
    """Summary of the complete integrated workflow."""
    modality_stats: Dict[str, IntegratedModalityStats] = field(default_factory=dict)
    item_results: List[IntegratedItemResult] = field(default_factory=list)
    execution_time: float = 0.0
    csv_report_path: Optional[Path] = None
    json_report_path: Optional[Path] = None

    @property
    def total_images(self) -> int:
        return sum(s.total_raw for s in self.modality_stats.values())

    @property
    def total_approved(self) -> int:
        return sum(s.approved for s in self.modality_stats.values())

    @property
    def total_rejected(self) -> int:
        return sum(s.rejected for s in self.modality_stats.values())

    @property
    def total_errors(self) -> int:
        return sum(s.errors for s in self.modality_stats.values())


class IntegratedPipeline:
    """
    Coordinator executing Phase 2 preprocessing and Phase 3 quality assessment sequentially.
    """

    def __init__(
        self,
        raw_dir: Optional[Union[str, Path]] = None,
        processed_dir: Optional[Union[str, Path]] = None,
        approved_dir: Optional[Union[str, Path]] = None,
        rejected_dir: Optional[Union[str, Path]] = None,
        log_dir: Optional[Union[str, Path]] = None,
        overwrite_phase2: bool = False,
    ):
        self.raw_dir = Path(raw_dir).resolve() if raw_dir else get_raw_dir()
        self.processed_dir = (
            Path(processed_dir).resolve() if processed_dir else get_processed_dir()
        )
        self.approved_dir = (
            Path(approved_dir).resolve() if approved_dir else get_approved_dir()
        )
        self.rejected_dir = (
            Path(rejected_dir).resolve() if rejected_dir else get_rejected_dir()
        )
        self.log_dir = Path(log_dir).resolve() if log_dir else get_integration_log_dir()
        self.overwrite_phase2 = overwrite_phase2

        # Ensure directory structures exist
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.approved_dir.mkdir(parents=True, exist_ok=True)
        self.rejected_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.log_dir / "integrated_pipeline_results.csv"
        self.json_path = self.log_dir / "integrated_pipeline_results.json"

    def process_image_item(
        self,
        raw_path: Path,
        modality: str,
        phase2_pipeline: PreprocessPipeline,
        phase3_pipeline: QualityAssessmentPipeline,
    ) -> IntegratedItemResult:
        """
        Process a single image item through Phase 2 -> Phase 3 -> Routing.
        """
        raw_name = raw_path.name
        processed_name = get_processed_filename(raw_path, output_ext="png")
        mod_processed_dir = self.processed_dir / modality
        mod_approved_dir = self.approved_dir / modality
        mod_rejected_dir = self.rejected_dir / modality

        mod_processed_dir.mkdir(parents=True, exist_ok=True)
        mod_approved_dir.mkdir(parents=True, exist_ok=True)
        mod_rejected_dir.mkdir(parents=True, exist_ok=True)

        processed_out_path = mod_processed_dir / processed_name

        # ----------------------------------------------------
        # STEP 1: EXECUTE PHASE 2 PREPROCESSING
        # ----------------------------------------------------
        phase2_success = False
        phase2_error: Optional[str] = None

        try:
            # If already processed and not overwriting, verify existence
            if not processed_out_path.exists() or self.overwrite_phase2:
                _, saved_path = phase2_pipeline.process(
                    input_path=raw_path, output_path=processed_out_path
                )
            phase2_success = True
        except Exception as e:
            phase2_error = f"{type(e).__name__}: {str(e)}"
            phase2_success = False

        if not phase2_success or not processed_out_path.exists():
            return IntegratedItemResult(
                raw_filename=raw_name,
                processed_filename=processed_name,
                modality=modality,
                phase2_status="FAILED",
                phase3_status="SKIPPED",
                overall_quality_score=None,
                decision="ERROR",
                is_approved_for_ai=False,
                routed_location="NONE",
                reason=f"Phase 2 preprocessing failed: {phase2_error}",
                error=phase2_error,
            )

        # ----------------------------------------------------
        # STEP 2: EXECUTE PHASE 3 QUALITY ASSESSMENT
        # ----------------------------------------------------
        phase3_res: Optional[AssessmentResult] = None
        phase3_error: Optional[str] = None

        try:
            phase3_res = phase3_pipeline.assess_file(processed_out_path)
        except Exception as e:
            phase3_error = f"{type(e).__name__}: {str(e)}"

        if phase3_res is None or phase3_error is not None:
            # Route to rejected due to Phase 3 failure
            dest_rejected = mod_rejected_dir / processed_name
            shutil.copy2(str(processed_out_path), str(dest_rejected))

            return IntegratedItemResult(
                raw_filename=raw_name,
                processed_filename=processed_name,
                modality=modality,
                phase2_status="SUCCESS",
                phase3_status="FAILED",
                overall_quality_score=None,
                decision="ERROR",
                is_approved_for_ai=False,
                routed_location=str(dest_rejected),
                reason=f"Phase 3 assessment failed: {phase3_error}",
                error=phase3_error,
            )

        # ----------------------------------------------------
        # STEP 3: ROUTE TO APPROVED OR REJECTED
        # ----------------------------------------------------
        if phase3_res.is_approved_for_ai:
            dest_path = mod_approved_dir / processed_name
            shutil.copy2(str(processed_out_path), str(dest_path))
            routed_loc = str(dest_path)
        else:
            dest_path = mod_rejected_dir / processed_name
            shutil.copy2(str(processed_out_path), str(dest_path))
            routed_loc = str(dest_path)

        return IntegratedItemResult(
            raw_filename=raw_name,
            processed_filename=processed_name,
            modality=modality,
            phase2_status="SUCCESS",
            phase3_status="SUCCESS",
            overall_quality_score=phase3_res.overall_score,
            decision=phase3_res.decision,
            is_approved_for_ai=phase3_res.is_approved_for_ai,
            routed_location=routed_loc,
            reason=phase3_res.reason,
            error=None,
        )

    def run(self, modality_filter: str = "all") -> IntegratedPipelineSummary:
        """
        Execute the integrated Phase 2 -> Phase 3 pipeline for specified modalities.

        Args:
            modality_filter: 'all', 'octa', 'octb', or 'fundus'.

        Returns:
            IntegratedPipelineSummary object.
        """
        start_time = time.time()
        filter_clean = modality_filter.strip().lower()

        supported = ("octa", "octb", "fundus")
        if filter_clean == "all":
            mods_to_run = list(supported)
        elif filter_clean in supported:
            mods_to_run = [filter_clean]
        else:
            raise ValueError(
                f"Unsupported modality '{modality_filter}'. Expected 'all' or one of: {supported}"
            )

        summary = IntegratedPipelineSummary(
            csv_report_path=self.csv_path,
            json_report_path=self.json_path,
        )

        for m in supported:
            summary.modality_stats[m] = IntegratedModalityStats()

        for mod in mods_to_run:
            mod_raw = self.raw_dir / mod
            if not mod_raw.exists():
                mod_raw.mkdir(parents=True, exist_ok=True)

            raw_files = find_image_files(mod_raw)
            st = summary.modality_stats[mod]
            st.total_raw = len(raw_files)

            if st.total_raw == 0:
                continue

            p2_pipeline = PreprocessPipeline(modality=mod)
            p3_pipeline = QualityAssessmentPipeline(modality=mod)

            for rf in raw_files:
                item_res = self.process_image_item(rf, mod, p2_pipeline, p3_pipeline)
                summary.item_results.append(item_res)

                if item_res.phase2_status == "SUCCESS":
                    st.phase2_success += 1
                else:
                    st.phase2_failed += 1

                if item_res.phase3_status == "SUCCESS":
                    st.phase3_assessed += 1

                if item_res.is_approved_for_ai:
                    st.approved += 1
                elif item_res.decision == "ERROR":
                    st.errors += 1
                else:
                    st.rejected += 1

        # Write reports
        self.write_reports(summary.item_results)

        summary.execution_time = round(time.time() - start_time, 2)
        return summary

    def write_reports(self, results: List[IntegratedItemResult]) -> None:
        """Write integrated run summaries to CSV and JSON."""
        # CSV
        fieldnames = [
            "raw_filename",
            "processed_filename",
            "modality",
            "phase2_status",
            "phase3_status",
            "overall_quality_score",
            "decision",
            "is_approved_for_ai",
            "routed_location",
            "reason",
            "error",
        ]
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(asdict(r))

        # JSON
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, indent=2)

    def format_summary(self, summary: IntegratedPipelineSummary) -> str:
        """Format the summary into human-readable report."""
        octa_st = summary.modality_stats.get("octa", IntegratedModalityStats())
        octb_st = summary.modality_stats.get("octb", IntegratedModalityStats())
        fundus_st = summary.modality_stats.get("fundus", IntegratedModalityStats())

        report = (
            "============================================================\n"
            "PHASE 2 -> PHASE 3 INTEGRATED PIPELINE SUMMARY\n"
            "============================================================\n\n"
            f"Total Raw Images Ingested : {summary.total_images}\n\n"
            "OCT-A\n"
            f"Raw Images      : {octa_st.total_raw}\n"
            f"Phase 2 Success : {octa_st.phase2_success}\n"
            f"Phase 2 Failed  : {octa_st.phase2_failed}\n"
            f"Phase 3 Assessed: {octa_st.phase3_assessed}\n"
            f"Approved for AI : {octa_st.approved}\n"
            f"Rejected        : {octa_st.rejected}\n"
            f"Pipeline Errors : {octa_st.errors}\n\n"
            "OCT-B\n"
            f"Raw Images      : {octb_st.total_raw}\n"
            f"Phase 2 Success : {octb_st.phase2_success}\n"
            f"Phase 2 Failed  : {octb_st.phase2_failed}\n"
            f"Phase 3 Assessed: {octb_st.phase3_assessed}\n"
            f"Approved for AI : {octb_st.approved}\n"
            f"Rejected        : {octb_st.rejected}\n"
            f"Pipeline Errors : {octb_st.errors}\n\n"
            "FUNDUS\n"
            f"Raw Images      : {fundus_st.total_raw}\n"
            f"Phase 2 Success : {fundus_st.phase2_success}\n"
            f"Phase 2 Failed  : {fundus_st.phase2_failed}\n"
            f"Phase 3 Assessed: {fundus_st.phase3_assessed}\n"
            f"Approved for AI : {fundus_st.approved}\n"
            f"Rejected        : {fundus_st.rejected}\n"
            f"Pipeline Errors : {fundus_st.errors}\n\n"
            "Overall\n"
            f"Total Images    : {summary.total_images}\n"
            f"Approved for AI : {summary.total_approved}\n"
            f"Rejected        : {summary.total_rejected}\n"
            f"Errors          : {summary.total_errors}\n\n"
            f"Execution Time  : {summary.execution_time:.2f} seconds\n\n"
            "Datasets Status:\n"
            f"- Approved Dataset (Ready for Phase 4): datasets/approved/\n"
            f"- Rejected Dataset                    : datasets/rejected/\n"
            f"- Integrated Log                      : {self.csv_path.name}\n\n"
            "============================================================"
        )
        return report


def run_phase2_phase3_pipeline(
    raw_dir: Optional[Union[str, Path]] = None,
    modality: str = "all",
    overwrite_phase2: bool = False,
) -> IntegratedPipelineSummary:
    """
    Public function to execute the full integrated pipeline.
    """
    pipeline = IntegratedPipeline(raw_dir=raw_dir, overwrite_phase2=overwrite_phase2)
    return pipeline.run(modality_filter=modality)


def main() -> None:
    """CLI entry point for integrated pipeline."""
    parser = argparse.ArgumentParser(
        description="Phase 2 -> Phase 3 Integrated Retinal AI Preprocessing & Quality Assessment"
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
        help="Path to raw dataset directory (defaults to datasets/raw/)",
    )
    parser.add_argument(
        "--overwrite-phase2",
        action="store_true",
        default=False,
        help="Force Phase 2 to reprocess existing processed files",
    )

    args = parser.parse_args()

    pipeline = IntegratedPipeline(
        raw_dir=args.raw_dir,
        overwrite_phase2=args.overwrite_phase2,
    )
    summary = pipeline.run(modality_filter=args.modality)
    print(pipeline.format_summary(summary))


if __name__ == "__main__":
    main()
