"""
Main Image Quality Assessment Pipeline for Retinal Imaging.

Coordinates the end-to-end technical evaluation:
1. Non-destructive Image Loading
2. Image Validation
3. Technical Metric Extraction (Blur, Brightness, Contrast, Noise, Clipping, Content, Color)
4. Modality-specific Metric Normalization (0-100)
5. Composite Quality Score Calculation
6. ACCEPT / WARNING / REJECT Decision Generation
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

from .config import (
    ModalityQualityConfig,
    get_modality_quality_config,
)
from .image_loader import load_image
from .validation import (
    validate_modality,
    validate_assessment_image,
)
from .blur_detection import compute_blur_metrics
from .brightness import compute_brightness_metrics
from .contrast import compute_contrast_metrics
from .noise import compute_noise_metrics
from .clipping import compute_clipping_metrics
from .content_quality import compute_content_metrics
from .color_quality import compute_color_quality_metrics
from .normalization import normalize_all_metrics
from .quality_score import calculate_overall_quality_score
from .decision import QualityDecision, make_decision


@dataclass
class AssessmentResult:
    """
    Structured result returned for every assessed retinal image.
    """
    image_name: str
    modality: str
    raw_metrics: Dict[str, Any]
    scores: Dict[str, float]
    overall_score: float
    decision: str
    is_approved_for_ai: bool
    reason: str
    failed_checks: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to a serializable dictionary."""
        return asdict(self)


class QualityAssessmentPipeline:
    """
    Modality-aware retinal image quality assessment pipeline.
    """

    def __init__(
        self,
        modality: str = "octa",
        config: Optional[ModalityQualityConfig] = None,
    ):
        """
        Initialize the QualityAssessmentPipeline.

        Args:
            modality: Modality identifier ('octa', 'octb', 'fundus').
            config: Optional custom ModalityQualityConfig.
        """
        self.modality = validate_modality(modality)
        self.config = config or get_modality_quality_config(self.modality)

    def assess_array(
        self,
        image: np.ndarray,
        image_name: str = "in_memory_image",
    ) -> AssessmentResult:
        """
        Assess technical quality on an in-memory numpy image array.

        Args:
            image: Input retinal image array.
            image_name: Optional identifier name.

        Returns:
            AssessmentResult dataclass.
        """
        # 1. Validate image
        validate_assessment_image(image, self.modality)

        # 2. Extract technical metrics non-destructively
        blur_res = compute_blur_metrics(image)
        bright_res = compute_brightness_metrics(image, is_color=self.config.is_color)
        contrast_res = compute_contrast_metrics(image)
        noise_res = compute_noise_metrics(image)
        clipping_res = compute_clipping_metrics(image, is_color=self.config.is_color)
        content_res = compute_content_metrics(image)
        color_res = compute_color_quality_metrics(image, is_color=self.config.is_color)

        raw_metrics: Dict[str, Any] = {
            **blur_res,
            **bright_res,
            **contrast_res,
            **noise_res,
            **clipping_res,
            **content_res,
            "color_metrics": color_res,
        }

        # 3. Normalize metrics to standardized 0-100 scale
        scores = normalize_all_metrics(raw_metrics, self.config)

        # 4. Calculate composite overall quality score
        overall_score = calculate_overall_quality_score(scores, self.config)

        # 5. Make quality decision
        decision_obj: QualityDecision = make_decision(overall_score, scores, self.config)

        return AssessmentResult(
            image_name=image_name,
            modality=self.modality,
            raw_metrics=raw_metrics,
            scores=scores,
            overall_score=round(overall_score, 2),
            decision=decision_obj.decision.value,
            is_approved_for_ai=decision_obj.is_approved_for_ai,
            reason=decision_obj.reason,
            failed_checks=decision_obj.failed_checks,
            error=None,
        )

    def assess_file(
        self,
        image_path: Union[str, Path],
    ) -> AssessmentResult:
        """
        Load and assess an image file from disk.

        Args:
            image_path: Path to the image file.

        Returns:
            AssessmentResult dataclass.
        """
        path = Path(image_path).resolve()
        # 1. Load image
        image = load_image(path, self.modality)
        # 2. Assess array
        return self.assess_array(image, image_name=path.name)


def assess_image(
    image: np.ndarray,
    modality: str = "octa",
    config: Optional[ModalityQualityConfig] = None,
    image_name: str = "in_memory_image",
) -> AssessmentResult:
    """
    High-level API to assess an in-memory retinal image array.

    Args:
        image: Numpy ndarray of the image.
        modality: Modality identifier ('octa', 'octb', 'fundus').
        config: Optional custom quality configuration.
        image_name: Name label for reporting.

    Returns:
        AssessmentResult dataclass.
    """
    pipeline = QualityAssessmentPipeline(modality=modality, config=config)
    return pipeline.assess_array(image=image, image_name=image_name)


def assess_image_file(
    image_path: Union[str, Path],
    modality: str = "octa",
    config: Optional[ModalityQualityConfig] = None,
) -> AssessmentResult:
    """
    High-level API to assess a retinal image file from disk.

    Args:
        image_path: Path to the image file.
        modality: Modality identifier ('octa', 'octb', 'fundus').
        config: Optional custom quality configuration.

    Returns:
        AssessmentResult dataclass.
    """
    pipeline = QualityAssessmentPipeline(modality=modality, config=config)
    return pipeline.assess_file(image_path=image_path)
