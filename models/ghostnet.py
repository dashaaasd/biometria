import math
import torch
import torch.nn as nn


class GhostModule(nn.Module):
    """
    Генерирует «ghost»-признаки: сначала обычная свёртка 1×1 (дорогая),
    затем дешёвая групповая свёртка 3×3 для создания «призрачных» копий.
    """
    def __init__(self, in_channels, out_channels, expansion_ratio=2, use_relu=True):
        super().__init__()

        primary_channels = math.ceil(out_channels / expansion_ratio)
        ghost_channels   = primary_channels * (expansion_ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, primary_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(primary_channels),
            nn.ReLU(inplace=True) if use_relu else nn.Identity()
        )

        self.ghost_conv = nn.Sequential(
            nn.Conv2d(primary_channels, ghost_channels,
                      kernel_size=3, stride=1, padding=1,
                      groups=primary_channels, bias=False),
            nn.BatchNorm2d(ghost_channels),
            nn.ReLU(inplace=True) if use_relu else nn.Identity()
        )

        self.out_channels = out_channels

    def forward(self, x):
        primary  = self.primary_conv(x)
        ghost    = self.ghost_conv(primary)
        out      = torch.cat([primary, ghost], dim=1)
        return out[:, :self.out_channels, :, :]


class GhostBottleneck(nn.Module):
    """
    Остаточный блок в стиле ResNet, где 1×1-свёртки заменены на GhostModule.
    Структура: расширение → (опциональный downsampling) → сжатие.
    """
    def __init__(self, in_channels, bottleneck_channels, out_channels, stride=1):
        super().__init__()

        self.expand_ghost = GhostModule(in_channels, bottleneck_channels, use_relu=True)

        self.downsample_dw = nn.Sequential(
            nn.Conv2d(bottleneck_channels, bottleneck_channels,
                      kernel_size=3, stride=stride, padding=1,
                      groups=bottleneck_channels, bias=False),
            nn.BatchNorm2d(bottleneck_channels)
        ) if stride > 1 else nn.Identity()

        self.project_ghost = GhostModule(bottleneck_channels, out_channels, use_relu=False)

        dimensions_match = (in_channels == out_channels) and (stride == 1)
        self.shortcut = nn.Identity() if dimensions_match else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.expand_ghost(x)
        x = self.downsample_dw(x)
        x = self.project_ghost(x)
        return x + residual
