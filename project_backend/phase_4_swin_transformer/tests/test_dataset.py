"""
Tests for RetinalDataset and DataLoader builders.
"""

from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch

from phase_4_swin_transformer.enums import Modality
from phase_4_swin_transformer.dataset import RetinalDataset, RetinalItem, create_dataloader


@pytest.fixture
def temp_dataset_dir(tmp_path: Path) -> Path:
    """Fixture creating mock folder structure."""
    class0_dir = tmp_path / "normal"
    class1_dir = tmp_path / "disease"
    class0_dir.mkdir(parents=True)
    class1_dir.mkdir(parents=True)

    # Write grayscale images for class 0
    for i in range(3):
        img = Image.fromarray(np.random.randint(50, 200, (64, 64), dtype=np.uint8))
        img.save(class0_dir / f"img_c0_{i}.png")

    # Write color RGB images for class 1
    for i in range(3):
        img = Image.fromarray(np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8))
        img.save(class1_dir / f"img_c1_{i}.png")

    return tmp_path


def test_dataset_from_folder_octa(temp_dataset_dir: Path):
    dataset = RetinalDataset.from_folder(temp_dataset_dir, modality=Modality.OCTA, image_size=224)
    assert len(dataset) == 6
    assert dataset.class_to_idx == {"disease": 0, "normal": 1}

    tensor_img, label, filename = dataset[0]
    assert isinstance(tensor_img, torch.Tensor)
    assert tensor_img.shape == (3, 224, 224)
    assert isinstance(label, int)
    assert isinstance(filename, str)


def test_dataset_from_folder_fundus(temp_dataset_dir: Path):
    dataset = RetinalDataset.from_folder(temp_dataset_dir, modality=Modality.FUNDUS, image_size=224)
    assert len(dataset) == 6
    tensor_img, label, _ = dataset[0]
    assert tensor_img.shape == (3, 224, 224)


def test_dataloader_creation(temp_dataset_dir: Path):
    dataset = RetinalDataset.from_folder(temp_dataset_dir, modality=Modality.OCTA, image_size=224)
    loader = create_dataloader(dataset, batch_size=2, shuffle=False)
    batch_imgs, batch_lbls, batch_names = next(iter(loader))

    assert batch_imgs.shape == (2, 3, 224, 224)
    assert batch_lbls.shape == (2,)
    assert len(batch_names) == 2
