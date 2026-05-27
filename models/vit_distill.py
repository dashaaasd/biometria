import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """Разбиение изображения на патчи и линейная проекция через Conv2d."""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=256):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.projection = nn.Conv2d(in_channels, embed_dim,
                                    kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.projection(x)   # [B, embed_dim, H/P, W/P]
        x = x.flatten(2)         # [B, embed_dim, num_patches]
        x = x.transpose(1, 2)    # [B, num_patches, embed_dim]
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT_Distill(nn.Module):
    """
    Vision Transformer с дистилляцией (DeiT-style).
    Добавлен dist_token и второй классификационный head.
    """
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 num_classes=7,
                 embed_dim=256,
                 depth=4,
                 num_heads=8,
                 mlp_ratio=4,
                 dropout=0.1):
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Два learnable токена
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head_cls = nn.Linear(embed_dim, num_classes)
        self.head_dist = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.dist_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, teacher_logits=None):
        B = x.shape[0]
        x = self.patch_embed(x)

        # Конкатенируем cls_token, dist_token и патчи
        tokens = torch.cat([
            self.cls_token.expand(B, -1, -1),
            self.dist_token.expand(B, -1, -1),
            x
        ], dim=1)

        tokens = self.pos_drop(tokens + self.pos_embed)

        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.norm(tokens)
        cls_out = tokens[:, 0]
        dist_out = tokens[:, 1]

        logits_cls = self.head_cls(cls_out)
        logits_dist = self.head_dist(dist_out)

        # Режим обучения с дистилляцией
        if teacher_logits is not None:
            teacher_targets = teacher_logits.argmax(dim=1).detach()
            loss_dist = F.cross_entropy(logits_dist, teacher_targets)
            loss_cls = F.cross_entropy(logits_cls, teacher_targets)
            return 0.5 * loss_cls + 0.5 * loss_dist

        # Инференс: усредняем два head'а (DeiT style)
        return (logits_cls + logits_dist) / 2