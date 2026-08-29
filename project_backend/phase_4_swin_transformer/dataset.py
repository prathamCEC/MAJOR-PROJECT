"""
Retinal Dataset Abstraction for Phase 4 Swin Transformer.

Provides modality-aware PyTorch Dataset implementations supporting both
folder structures and metadata CSV manifests with patient ID tracking.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from .enums import Modality
from .transforms import get_transforms


@dataclass
class RetinalItem:
    """Single item metadata in Retinal Dataset."""
    image_path: Path
    modality: Modality
    label: int
    class_name: str
    patient_id: Optional[str] = None


class RetinalDataset(Dataset):
    """
    Modality-aware PyTorch Dataset for Retinal Imaging.
    """

    def __init__(
        self,
        items: List[RetinalItem],
        modality: Union[str, Modality],
        transform: Optional[Callable] = None,
        is_training: bool = False,
        image_size: int = 224,
    ):
        """
        Initialize RetinalDataset.

        Args:
            items: List of RetinalItem records.
            modality: Modality enum or string.
            transform: Optional custom transform pipeline.
            is_training: Whether dataset is used for training (applies augmentations if transform=None).
            image_size: Target image dimensions.
        """
        self.items = items
        self.modality = Modality.from_str(modality) if isinstance(modality, str) else modality
        self.is_training = is_training
        self.image_size = image_size
        self.transform = transform or get_transforms(
            modality=self.modality,
            is_training=self.is_training,
            image_size=self.image_size,
        )

        # Build class index mapping
        unique_classes = sorted(list({item.class_name for item in self.items}))
        self.class_to_idx: Dict[str, int] = {c: i for i, c in enumerate(unique_classes)}
        self.idx_to_class: Dict[int, str] = {i: c for c, i in self.class_to_idx.items()}

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        item = self.items[idx]
        
        # Load image via PIL non-destructively
        if not item.image_path.exists():
            raise FileNotFoundError(f"Image not found at path: {item.image_path}")

        try:
            with Image.open(item.image_path) as img:
                # Fundus: Ensure true RGB; OCT: Grayscale mode handled by transform adapter
                if self.modality == Modality.FUNDUS:
                    image = img.convert("RGB")
                else:
                    image = img.convert("L")
        except Exception as e:
            raise RuntimeError(f"Error loading image '{item.image_path}': {e}") from e

        if self.transform:
            tensor_image = self.transform(image)
        else:
            tensor_image = T.ToTensor()(image)

        return tensor_image, item.label, item.image_path.name

    @classmethod
    def from_folder(
        cls,
        root_dir: Union[str, Path],
        modality: Union[str, Modality],
        is_training: bool = False,
        image_size: int = 224,
        transform: Optional[Callable] = None,
    ) -> "RetinalDataset":
        """
        Build dataset from class-subfolder directory structure:
        root_dir/
            class_0/
                img1.png
            class_1/
                img2.png
        """
        path = Path(root_dir).resolve()
        mod_enum = Modality.from_str(modality) if isinstance(modality, str) else modality

        if not path.exists():
            raise FileNotFoundError(f"Dataset root directory does not exist: {path}")

        valid_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".ppm"}
        items: List[RetinalItem] = []

        # Find subdirectories representing classes
        subdirs = [d for d in path.iterdir() if d.is_dir()]
        subdirs.sort(key=lambda d: d.name)

        if not subdirs:
            # Check if flat directory without subfolders (e.g. single unclassified folder)
            flat_files = [f for f in path.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions]
            for f in flat_files:
                items.append(
                    RetinalItem(
                        image_path=f,
                        modality=mod_enum,
                        label=0,
                        class_name="unlabeled",
                        patient_id=None,
                    )
                )
        else:
            for label_idx, class_dir in enumerate(subdirs):
                class_name = class_dir.name
                for file_path in class_dir.iterdir():
                    if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
                        items.append(
                            RetinalItem(
                                image_path=file_path,
                                modality=mod_enum,
                                label=label_idx,
                                class_name=class_name,
                                patient_id=None,
                            )
                        )

        return cls(
            items=items,
            modality=mod_enum,
            transform=transform,
            is_training=is_training,
            image_size=image_size,
        )

    @classmethod
    def from_csv(
        cls,
        csv_path: Union[str, Path],
        modality: Union[str, Modality],
        is_training: bool = False,
        image_size: int = 224,
        transform: Optional[Callable] = None,
    ) -> "RetinalDataset":
        """
        Build dataset from CSV manifest containing:
        image_path,modality,label,class_name[,patient_id]
        """
        path = Path(csv_path).resolve()
        mod_enum = Modality.from_str(modality) if isinstance(modality, str) else modality

        if not path.exists():
            raise FileNotFoundError(f"Manifest CSV does not exist: {path}")

        df = pd.read_csv(path)
        items: List[RetinalItem] = []

        for _, row in df.iterrows():
            img_p = Path(str(row["image_path"]))
            if not img_p.is_absolute():
                img_p = (path.parent / img_p).resolve()

            label_val = int(row["label"])
            class_name = str(row.get("class_name", f"class_{label_val}"))
            patient_id = str(row["patient_id"]) if "patient_id" in row and pd.notna(row["patient_id"]) else None

            items.append(
                RetinalItem(
                    image_path=img_p,
                    modality=mod_enum,
                    label=label_val,
                    class_name=class_name,
                    patient_id=patient_id,
                )
            )

        return cls(
            items=items,
            modality=mod_enum,
            transform=transform,
            is_training=is_training,
            image_size=image_size,
        )


def create_dataloader(
    dataset: RetinalDataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = False,
) -> DataLoader:
    """
    Factory creating a robust DataLoader with Windows compatibility.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
    )
