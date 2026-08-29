"""
Modality-Aware Transforms for Swin Transformer.

Provides modality-specific data augmentation for training and deterministic
preprocessing for validation/testing.
"""

from typing import Tuple, Union
import torch
import torchvision.transforms as T
from PIL import Image

from .enums import Modality


class GrayscaleTo3Channels:
    """
    Adapter converting 1-channel grayscale PIL image or tensor to 3 identical channels
    required by standard Swin Transformer backbones.
    Does NOT fabricate color; documents channel replication.
    """
    def __call__(self, img: Image.Image) -> Image.Image:
        if img.mode != "RGB":
            return img.convert("RGB")
        return img


def get_transforms(
    modality: Union[str, Modality],
    is_training: bool = False,
    image_size: int = 224,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> T.Compose:
    """
    Build modality-aware PyTorch transform pipeline.

    Args:
        modality: Modality enum or string ('octa', 'octb', 'fundus').
        is_training: If True, applies safe conservative augmentations.
        image_size: Target resolution (default 224).
        mean: Normalization mean tuple.
        std: Normalization standard deviation tuple.

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    mod_enum = Modality.from_str(modality) if isinstance(modality, str) else modality
    transform_list = []

    # 1. Channel adaptation & initial sizing
    if mod_enum in (Modality.OCTA, Modality.OCTB):
        # Grayscale modalities converted cleanly to 3 replicated channels for Swin
        transform_list.append(GrayscaleTo3Channels())
    else:
        # Fundus photography preserves true RGB
        transform_list.append(GrayscaleTo3Channels())

    # 2. Augmentations (Training only)
    if is_training:
        if mod_enum == Modality.OCTA:
            # OCT-A: Capillary microvasculature preservation
            transform_list.extend([
                T.Resize((image_size, image_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomRotation(degrees=10),
            ])
        elif mod_enum == Modality.OCTB:
            # OCT-B: Layer cross-sections (horizontal flips and subtle brightness)
            transform_list.extend([
                T.Resize((image_size, image_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.1, contrast=0.1),
            ])
        elif mod_enum == Modality.FUNDUS:
            # Fundus: Color photographs (mild affine & color dynamics)
            transform_list.extend([
                T.Resize((image_size, image_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomRotation(degrees=15),
                T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            ])
    else:
        # Deterministic evaluation transforms
        transform_list.append(T.Resize((image_size, image_size)))

    # 3. Conversion to Tensor & Normalization
    transform_list.extend([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    return T.Compose(transform_list)
