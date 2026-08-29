"""
Tests for modality-aware transform pipelines.
"""

from PIL import Image
import numpy as np
import torch
import pytest

from phase_4_swin_transformer.enums import Modality
from phase_4_swin_transformer.transforms import get_transforms, GrayscaleTo3Channels


def test_grayscale_to_3channels_adapter():
    adapter = GrayscaleTo3Channels()
    gray_img = Image.fromarray(np.ones((64, 64), dtype=np.uint8) * 128, mode="L")
    rgb_img = adapter(gray_img)

    assert rgb_img.mode == "RGB"
    arr = np.array(rgb_img)
    assert arr.shape == (64, 64, 3)
    assert np.all(arr[:, :, 0] == arr[:, :, 1])
    assert np.all(arr[:, :, 1] == arr[:, :, 2])


def test_training_transforms_octa():
    t = get_transforms(Modality.OCTA, is_training=True, image_size=224)
    gray_img = Image.fromarray(np.random.randint(0, 255, (100, 100), dtype=np.uint8), mode="L")
    tensor = t(gray_img)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_eval_transforms_deterministic():
    t = get_transforms(Modality.FUNDUS, is_training=False, image_size=224)
    rgb_img = Image.fromarray(np.random.randint(0, 255, (150, 150, 3), dtype=np.uint8), mode="RGB")
    tensor1 = t(rgb_img)
    tensor2 = t(rgb_img)

    assert torch.allclose(tensor1, tensor2, atol=1e-5)
