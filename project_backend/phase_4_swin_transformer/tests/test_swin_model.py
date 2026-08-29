"""
Model sanity tests: instantiation, forward pass, loss calculation, and backward gradients.
"""

import pytest
import torch
import torch.nn as nn

from phase_4_swin_transformer.enums import Modality
from phase_4_swin_transformer.models.swin_factory import create_swin_model
from phase_4_swin_transformer.models.octa_model import OCTASwinModel
from phase_4_swin_transformer.models.octb_model import OCTBSwinModel
from phase_4_swin_transformer.models.fundus_model import FundusSwinModel


@pytest.mark.parametrize("modality", [Modality.OCTA, Modality.OCTB, Modality.FUNDUS])
def test_swin_model_sanity_and_gradients(modality: Modality):
    """
    Sanity test for each modality:
    1. Create model
    2. Forward pass with batch
    3. Calculate loss
    4. Backward pass & verify gradients exist
    """
    num_classes = 2
    model = create_swin_model(
        modality=modality,
        num_classes=num_classes,
        pretrained=False,
    )

    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    dummy_target = torch.tensor([0, 1], dtype=torch.long)

    # Forward pass
    output = model(dummy_input)
    assert output.shape == (batch_size, num_classes)
    assert torch.isfinite(output).all()

    # Loss calculation
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, dummy_target)
    assert torch.isfinite(loss)
    assert loss.item() > 0

    # Backward pass
    loss.backward()

    # Verify gradients exist in trainable parameters
    has_grad = False
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            has_grad = True
            assert torch.isfinite(p.grad).all()
            break

    assert has_grad, f"Gradients were not computed for {modality.value} model"


def test_specialized_model_wrappers():
    octa_m = OCTASwinModel(num_classes=2, pretrained=False)
    octb_m = OCTBSwinModel(num_classes=3, pretrained=False)
    fundus_m = FundusSwinModel(num_classes=2, pretrained=False)

    x = torch.randn(1, 3, 224, 224)
    assert octa_m(x).shape == (1, 2)
    assert octb_m(x).shape == (1, 3)
    assert fundus_m(x).shape == (1, 2)


def test_backbone_freezing():
    model = create_swin_model(Modality.OCTA, num_classes=2, pretrained=False, freeze_backbone=True)
    # Check that backbone params do not require grad
    for name, param in model.backbone.named_parameters():
        if "head" not in name and "fc" not in name and "classifier" not in name:
            assert not param.requires_grad

    model.unfreeze_backbone()
    for param in model.backbone.parameters():
        assert param.requires_grad
