import torch.nn as nn
import torch.nn.functional as F

from ghostnet import GhostBottleneck


class GhostNetV2_FER(nn.Module):
    """
    GhostNetV2-lite для распознавания эмоций.

    Поддерживает два режима разрешения:
      img_size=48  — FER2013, ExpW_48, GFFD_48   (patch 48→24→12→6)
      img_size=224 — ExpW, GFFD-2025             (patch 224→112→56→28→14)

    AdaptiveAvgPool2d(1) в head убирает зависимость от финального
    пространственного размера, поэтому архитектура одна для всех датасетов.
    """
    def __init__(self, num_classes=7, dropout=0.3, img_size=48, in_channels=1):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        # Три этапа downsampling, общие для всех разрешений
        self.stage1 = GhostBottleneck(16,  48,  24, stride=2)   # H/2
        self.stage2 = GhostBottleneck(24,  72,  40, stride=2)   # H/4
        self.stage3 = GhostBottleneck(40, 120,  80, stride=2)   # H/8

        # Для 224px добавляем ещё один stride=2, чтобы карта была ~14×14
        # вместо 28×28 перед финальными блоками
        if img_size >= 112:
            self.stage4 = GhostBottleneck(80, 160, 80, stride=2)  # H/16
        else:
            self.stage4 = nn.Identity()

        self.blocks = nn.Sequential(
            GhostBottleneck(80,  160,  80, stride=1),
            GhostBottleneck(80,  160,  80, stride=1),
            GhostBottleneck(80,  240, 112, stride=1),
            GhostBottleneck(112, 336, 112, stride=1),
            GhostBottleneck(112, 480, 160, stride=1),
        )

        # AdaptiveAvgPool2d(1) → вектор 960 независимо от разрешения
        self.head = nn.Sequential(
            nn.Conv2d(160, 960, kernel_size=1, bias=False),
            nn.BatchNorm2d(960),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc         = nn.Linear(960, 128)
        self.bn_fc      = nn.BatchNorm1d(128)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(128, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.blocks(x)
        x = self.head(x).flatten(1)   # [B, 960]
        x = self.fc(x)                # [B, 128]
        x = self.bn_fc(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.classifier(x)        # [B, num_classes]
        return x
