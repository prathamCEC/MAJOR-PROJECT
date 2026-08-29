"""
Tests for Feature Tokenization module.
"""

import pytest
import torch

from phase_6_clinical_transformer.feature_tokenizer import (
    NumericalFeatureTokenizer,
    CategoricalFeatureTokenizer,
    ClinicalFeatureTokenizer,
)


def test_numerical_tokenizer():
    tokenizer = NumericalFeatureTokenizer(num_numerical=3, embed_dim=256)
    x_num = torch.randn(4, 3)
    tokens = tokenizer(x_num)
    assert tokens.shape == (4, 3, 256)
    assert torch.isfinite(tokens).all()


def test_categorical_tokenizer():
    cardinalities = [3, 4, 2]  # 3 categorical features
    tokenizer = CategoricalFeatureTokenizer(cardinalities=cardinalities, embed_dim=256)
    x_cat = torch.tensor([
        [0, 1, 0],
        [2, 3, 1],
    ], dtype=torch.long)

    tokens = tokenizer(x_cat)
    assert tokens.shape == (2, 3, 256)
    assert torch.isfinite(tokens).all()


def test_clinical_feature_tokenizer_with_cls():
    num_num = 2
    cards = [3, 4]
    tokenizer = ClinicalFeatureTokenizer(
        num_numerical=num_num,
        categorical_cardinalities=cards,
        embed_dim=256,
    )
    x_num = torch.randn(5, num_num)
    x_cat = torch.tensor([[1, 2], [0, 1], [2, 3], [1, 0], [0, 0]], dtype=torch.long)

    # 1 ([CLS]) + 2 (num) + 2 (cat) = 5 tokens
    tokens = tokenizer(x_num, x_cat)
    assert tokens.shape == (5, 5, 256)
    assert torch.isfinite(tokens).all()
