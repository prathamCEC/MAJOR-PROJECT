"""
Modular Pipeline for Phase 2 Retinal Image Preprocessing.

Coordinates all preprocessing steps in strict sequence:
1. Loading & initial file check
2. Raw image validation
3. Safe black border detection & cropping
4. Modality-specific CLAHE contrast enhancement
5. Conservative Gaussian smoothing
6. Modality-specific Median filtering
7. Pixel range normalization & uint8 quantization
8. Aspect-ratio-preserving resize & symmetric padding
9. Channel standardization (3 channels for Swin Transformer)
10. Post-processing dimensional & statistical validation
11. Atomic disk saving and disk-reload verification
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np

from .config import (
    ModalityConfig,
    get_modality_config,
    TARGET_WIDTH,
    TARGET_HEIGHT,
)
from .image_loader import load_image
from .validation import (
    validate_modality,
    validate_raw_image,
    validate_processed_image,
)
from .border_crop import detect_and_crop_borders
from .clahe import apply_clahe
from .gaussian_filter import apply_gaussian_filter
from .median_filter import apply_median_filter
from .normalization import normalize_to_float32, convert_to_uint8
from .resizing import resize_with_aspect_ratio_and_pad
from .utils import (
    ensure_three_channels,
    save_processed_image,
    get_processed_filename,
)


class PreprocessPipeline:
    """
    Modality-aware retinal image preprocessing pipeline.

    Encapsulates all preprocessing steps for a single modality or custom configuration.
    """

    def __init__(
        self,
        modality: str = "octa",
        config: Optional[ModalityConfig] = None,
    ):
        """
        Initialize the PreprocessPipeline.

        Args:
            modality: Imaging modality identifier ('octa', 'octb', 'fundus').
            config: Optional custom ModalityConfig. If None, loaded from defaults.
        """
        self.modality = validate_modality(modality)
        self.config = config or get_modality_config(self.modality)

    def process_array(self, image: np.ndarray) -> np.ndarray:
        """
        Execute the full preprocessing pipeline on an in-memory numpy image array.

        Args:
            image: Raw input numpy ndarray.

        Returns:
            Preprocessed numpy ndarray with shape (224, 224, 3) and dtype uint8.
        """
        # 1. Validate raw in-memory image
        validate_raw_image(image, self.modality)
        current = image.copy()

        # 2. Safe border detection and cropping
        if self.config.apply_border_crop:
            current = detect_and_crop_borders(
                current,
                threshold=self.config.crop_threshold,
                margin=self.config.crop_margin,
                min_border_ratio=self.config.min_border_ratio,
            )

        # 3. Contrast enhancement via CLAHE
        if self.config.apply_clahe:
            current = apply_clahe(
                current,
                clip_limit=self.config.clahe_clip_limit,
                tile_grid_size=self.config.clahe_tile_grid_size,
                is_color=self.config.is_color,
            )

        # 4. Conservative Gaussian smoothing
        if self.config.apply_gaussian:
            current = apply_gaussian_filter(
                current,
                kernel_size=self.config.gaussian_kernel_size,
                sigma=self.config.gaussian_sigma,
            )

        # 5. Modality-specific Median filtering (e.g. for OCT-B speckle noise)
        if self.config.apply_median:
            current = apply_median_filter(
                current,
                kernel_size=self.config.median_kernel_size,
            )

        # 6. Normalization: internal float32 conversion & uint8 quantization
        float_img = normalize_to_float32(
            current,
            target_min=self.config.norm_min,
            target_max=self.config.norm_max,
        )
        current = convert_to_uint8(
            float_img,
            source_min=self.config.norm_min,
            source_max=self.config.norm_max,
        )

        # 7. Aspect-ratio-preserving resize and symmetric padding to exact target size
        current = resize_with_aspect_ratio_and_pad(
            current,
            target_size=self.config.target_size,
            pad_value=self.config.pad_value,
            interpolation_down=self.config.resize_interpolation_down,
            interpolation_up=self.config.resize_interpolation_up,
        )

        # 8. Standardize to 3 channels for Swin Transformer compatibility
        if self.config.convert_to_3_channel:
            current = ensure_three_channels(current)

        # 9. Final post-processing validation
        validate_processed_image(
            current,
            target_size=self.config.target_size,
            expected_channels=3 if self.config.convert_to_3_channel else (3 if self.config.is_color else 1),
        )

        return current

    def process(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
    ) -> Tuple[np.ndarray, Optional[Path]]:
        """
        Load, preprocess, and optionally save a retinal image file.

        Args:
            input_path: Path to the raw image file.
            output_path: Optional destination file path.

        Returns:
            Tuple of (preprocessed_image_array, saved_path_or_None).
        """
        # Step 1: Load image in a modality-aware manner
        raw_image = load_image(input_path, self.modality)

        # Step 2-9: Execute preprocessing sequence
        processed_image = self.process_array(raw_image)

        # Step 10: Save to disk and verify reload if output_path is requested
        saved_file_path: Optional[Path] = None
        if output_path is not None:
            saved_file_path = save_processed_image(
                image=processed_image,
                output_path=output_path,
                expected_shape=self.config.target_size,
            )

        return processed_image, saved_file_path


def preprocess_image(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    modality: str = "octa",
    config: Optional[ModalityConfig] = None,
) -> Tuple[np.ndarray, Optional[Path]]:
    """
    High-level, stable public API to preprocess a single retinal image.
    Designed for clean consumption by Phase 3 (Image Quality Assessment) and beyond.

    Args:
        input_path: Path to the input image file.
        output_path: Optional destination path to write the processed image.
        modality: Modality identifier ('octa', 'octb', 'fundus').
        config: Optional custom configuration object.

    Returns:
        Tuple of (preprocessed_numpy_array, resolved_output_path_or_None).
    """
    pipeline = PreprocessPipeline(modality=modality, config=config)
    return pipeline.process(input_path=input_path, output_path=output_path)
