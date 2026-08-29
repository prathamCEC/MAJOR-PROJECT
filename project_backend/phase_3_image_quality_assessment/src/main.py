"""
Command Line Interface for Phase 3 Retinal Image Quality Assessment.

Supports both single-image technical evaluation and dataset batch assessment.
"""

import argparse
import json
from pathlib import Path
import sys

from .config import SUPPORTED_MODALITIES
from .pipeline import QualityAssessmentPipeline, assess_image_file
from .batch_processor import Phase3BatchProcessor


def main() -> None:
    """CLI entry point for Phase 3."""
    parser = argparse.ArgumentParser(
        description="Phase 3 — Retinal Image Quality Assessment"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to a single image file to assess",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to directory containing processed images (defaults to datasets/processed/)",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="all",
        choices=["all", "octa", "octb", "fundus"],
        help="Imaging modality: 'all' (default), 'octa', 'octb', or 'fundus'",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory to save CSV/JSON reports (defaults to logs/)",
    )

    args = parser.parse_args()

    # Single Image Mode
    if args.image:
        if args.modality == "all":
            print("Error: When evaluating a single image, you must specify a specific --modality (octa, octb, fundus).")
            sys.exit(1)

        img_path = Path(args.image).resolve()
        if not img_path.exists():
            print(f"Error: Image file not found: {img_path}")
            sys.exit(1)

        try:
            result = assess_image_file(image_path=img_path, modality=args.modality)
            print("\n============================================================")
            print("PHASE 3 QUALITY ASSESSMENT RESULT (SINGLE IMAGE)")
            print("============================================================")
            print(f"Image Name    : {result.image_name}")
            print(f"Modality      : {result.modality.upper()}")
            print(f"Overall Score : {result.overall_score:.2f} / 100.0")
            print(f"Decision      : {result.decision}")
            print(f"AI Approved   : {result.is_approved_for_ai}")
            print(f"Reason        : {result.reason}")
            print("\nDimension Scores (0 - 100):")
            for metric, score in sorted(result.scores.items()):
                print(f"  - {metric:<18}: {score:6.2f}")
            if result.failed_checks:
                print("\nFailed Checks:")
                for fc in result.failed_checks:
                    print(f"  * {fc}")
            print("============================================================\n")
        except Exception as e:
            print(f"Error during image assessment: {e}")
            sys.exit(1)
        return

    # Batch Mode
    processor = Phase3BatchProcessor(
        input_dir=args.input,
        log_dir=args.log_dir,
    )
    summary = processor.run(modality_filter=args.modality)
    print(processor.format_summary(summary))


if __name__ == "__main__":
    main()
