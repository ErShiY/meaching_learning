"""Reusable spatial aggregation and multi-label classification heads."""

import torch
import torch.nn as nn


class LearnableGeM(nn.Module):
    """Generalized mean over nonnegative spatial activations, with learned p."""

    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p)))
        self.eps = eps

    def forward(self, features):
        # Bounded exponents and float32 avoid unstable powers at low precision.
        exponent = self.p.clamp(1.0, 6.0)
        features = features.float().clamp_min(self.eps)
        return features.pow(exponent).mean(dim=(-2, -1)).pow(1.0 / exponent)


class LabelAttentionHead(nn.Module):
    """Use one learned query per label to pool multi-scale spatial evidence.

    Queries attend to spatial positions, not to competing label probabilities.
    A global branch supplements local evidence. Default dimensions and parameter
    names match the original resnet18_fusion head for checkpoint compatibility.
    """

    def __init__(
        self, num_classes, hidden_dim=256, dropout=0.2,
        mid_channels=256, deep_channels=512, global_pooling="avgmax",
    ):
        super().__init__()
        if global_pooling not in ("avgmax", "gem"):
            raise ValueError(f"Unknown global pooling: {global_pooling}")
        self.global_pooling = global_pooling
        self.mid_projection = nn.Sequential(
            nn.Conv2d(mid_channels, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(32, hidden_dim),
            nn.GELU(),
        )
        self.deep_projection = nn.Sequential(
            nn.Conv2d(deep_channels, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(32, hidden_dim),
            nn.GELU(),
        )
        self.scale_embeddings = nn.Parameter(torch.empty(2, hidden_dim))
        self.label_queries = nn.Parameter(torch.empty(num_classes, hidden_dim))
        nn.init.normal_(self.scale_embeddings, std=0.02)
        nn.init.normal_(self.label_queries, std=0.02)

        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=4, dropout=0.1, batch_first=True
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.label_weight = nn.Parameter(torch.empty(num_classes, hidden_dim))
        self.label_bias = nn.Parameter(torch.zeros(num_classes))
        nn.init.normal_(self.label_weight, std=0.02)

        if global_pooling == "gem":
            self.global_pool = LearnableGeM()
            global_dim = deep_channels
        else:
            global_dim = 2 * deep_channels
        self.global_head = nn.Sequential(
            nn.LayerNorm(global_dim),
            nn.Linear(global_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.label_residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, mid_features, deep_features):
        mid_tokens = self.mid_projection(mid_features).flatten(2).transpose(1, 2)
        deep_tokens = self.deep_projection(deep_features).flatten(2).transpose(1, 2)
        memory = torch.cat(
            [
                mid_tokens + self.scale_embeddings[0],
                deep_tokens + self.scale_embeddings[1],
            ],
            dim=1,
        )
        memory = self.memory_norm(memory)
        queries = self.label_queries.unsqueeze(0).expand(memory.size(0), -1, -1)
        attended, _ = self.cross_attention(
            self.query_norm(queries), memory, memory, need_weights=False
        )
        label_features = self.attention_norm(
            queries + self.attention_dropout(attended)
        )
        label_features = self.output_norm(
            label_features + self.feed_forward(label_features)
        )
        label_logits = torch.einsum(
            "bkd,kd->bk", label_features, self.label_weight
        ) + self.label_bias

        if self.global_pooling == "gem":
            global_features = self.global_pool(deep_features)
        else:
            global_features = torch.cat(
                [deep_features.mean(dim=(-2, -1)), deep_features.amax(dim=(-2, -1))],
                dim=1,
            )
        return self.global_head(global_features) + (
            self.label_residual_scale * label_logits
        )
