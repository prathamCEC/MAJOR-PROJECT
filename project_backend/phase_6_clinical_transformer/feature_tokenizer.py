"""
Feature Tokenization Module for FT-Transformer.

Implements standard FT-Transformer feature-level tokenization (Gorishniy et al., NeurIPS 2021)
transforming numerical and categorical clinical attributes into learnable token embeddings.
"""

import math
from typing import Dict, List, Optional
import torch
import torch.nn as nn


class NumericalFeatureTokenizer(nn.Module):
    """
    Transforms continuous numerical scalar features into D-dimensional token embeddings.
    
    Each numerical feature j is projected via a dedicated learnable parameter vector W_j and bias b_j:
    e_j = x_j * W_j + b_j
    """

    def __init__(self, num_numerical: int, embed_dim: int = 256):
        super().__init__()
        self.num_numerical = num_numerical
        self.embed_dim = embed_dim

        if num_numerical > 0:
            # Dedicated weights for each numerical feature [num_numerical, embed_dim]
            self.weights = nn.Parameter(torch.empty(num_numerical, embed_dim))
            self.biases = nn.Parameter(torch.empty(num_numerical, embed_dim))
            self._init_weights()
        else:
            self.weights = None
            self.biases = None

    def _init_weights(self) -> None:
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        nn.init.zeros_(self.biases)

    def forward(self, x_num: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_num: Float tensor of shape [B, num_numerical]

        Returns:
            Token tensor of shape [B, num_numerical, embed_dim]
        """
        if self.num_numerical == 0 or x_num.shape[-1] == 0:
            B = x_num.shape[0]
            return torch.empty(B, 0, self.embed_dim, device=x_num.device)

        # Broadcast multiply: [B, num_numerical, 1] * [1, num_numerical, embed_dim] + [1, num_numerical, embed_dim]
        tokens = x_num.unsqueeze(-1) * self.weights.unsqueeze(0) + self.biases.unsqueeze(0)
        return tokens


class CategoricalFeatureTokenizer(nn.Module):
    """
    Transforms discrete categorical indices into D-dimensional token embeddings.
    """

    def __init__(self, cardinalities: List[int], embed_dim: int = 256):
        super().__init__()
        self.cardinalities = cardinalities
        self.embed_dim = embed_dim

        if cardinalities:
            # Embedding tables for each categorical feature
            self.embeddings = nn.ModuleList([
                nn.Embedding(num_embeddings=card, embedding_dim=embed_dim)
                for card in cardinalities
            ])
            self._init_embeddings()
        else:
            self.embeddings = nn.ModuleList()

    def _init_embeddings(self) -> None:
        for emb in self.embeddings:
            nn.init.normal_(emb.weight, std=0.02)

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_cat: Long tensor of shape [B, num_categorical]

        Returns:
            Token tensor of shape [B, num_categorical, embed_dim]
        """
        if not self.cardinalities or x_cat.shape[-1] == 0:
            B = x_cat.shape[0]
            return torch.empty(B, 0, self.embed_dim, device=x_cat.device)

        token_list = []
        for j, emb in enumerate(self.embeddings):
            col_indices = x_cat[:, j]  # [B]
            col_tokens = emb(col_indices).unsqueeze(1)  # [B, 1, embed_dim]
            token_list.append(col_tokens)

        # Concatenate along feature dimension -> [B, num_categorical, embed_dim]
        return torch.cat(token_list, dim=1)


class ClinicalFeatureTokenizer(nn.Module):
    """
    Unified Tokenizer for Clinical Data.

    Generates numerical and categorical tokens, concatenates them, and prepends
    a learnable [CLS] token for downstream representation pooling.
    """

    def __init__(
        self,
        num_numerical: int,
        categorical_cardinalities: List[int],
        embed_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_numerical = num_numerical
        self.categorical_cardinalities = categorical_cardinalities
        self.embed_dim = embed_dim

        # Numerical and Categorical tokenizers
        self.num_tokenizer = NumericalFeatureTokenizer(num_numerical=num_numerical, embed_dim=embed_dim)
        self.cat_tokenizer = CategoricalFeatureTokenizer(cardinalities=categorical_cardinalities, embed_dim=embed_dim)

        # Learnable [CLS] token [1, 1, embed_dim]
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Normalization and dropout
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x_num: torch.Tensor,
        x_cat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_num: Float tensor [B, num_numerical]
            x_cat: Long tensor [B, num_categorical]

        Returns:
            Tokens tensor including [CLS] of shape [B, 1 + num_features, embed_dim]
        """
        B = x_num.shape[0] if x_num.ndim > 0 and x_num.shape[0] > 0 else x_cat.shape[0]

        token_streams = []

        # 1. Numerical tokens [B, N_num, D]
        if self.num_numerical > 0 and x_num.shape[-1] > 0:
            num_tokens = self.num_tokenizer(x_num)
            token_streams.append(num_tokens)

        # 2. Categorical tokens [B, N_cat, D]
        if self.categorical_cardinalities and x_cat.shape[-1] > 0:
            cat_tokens = self.cat_tokenizer(x_cat)
            token_streams.append(cat_tokens)

        if token_streams:
            feature_tokens = torch.cat(token_streams, dim=1)
        else:
            feature_tokens = torch.empty(B, 0, self.embed_dim, device=x_num.device)

        # 3. Expand and prepend [CLS] token
        cls_expanded = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        all_tokens = torch.cat([cls_expanded, feature_tokens], dim=1)  # [B, 1 + N_feat, embed_dim]

        return self.dropout(self.norm(all_tokens))
