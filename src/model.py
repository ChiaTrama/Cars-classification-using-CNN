import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class SimpleResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, 
                                  downsample=nn.Sequential(
                                      nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                                      nn.BatchNorm2d(out_channels)
                                  ) if stride != 1 or in_channels != out_channels else None))
        
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn(self.conv(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class SimpleResNetLarge(nn.Module):
    def __init__(self, num_classes=75):
        super().__init__()
        self.conv = nn.Conv2d(3, 40, 7, 2, 3, bias=False)
        self.bn = nn.BatchNorm2d(40)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(40, 80, 3, stride=1)
        self.layer2 = self._make_layer(80, 160, 4, stride=2)
        self.layer3 = self._make_layer(160, 320, 6, stride=2)
        self.layer4 = self._make_layer(320, 640, 3, stride=2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(640, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualBlock(
            in_channels, out_channels, stride,
            downsample=nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            ) if stride != 1 or in_channels != out_channels else None
        ))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn(self.conv(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# Fine-tuning ResNet18
def get_resnet18(num_classes=10, pretrained=True):
    model = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
    in_features = model.fc.in_features  # 512
    model.fc = nn.Linear(in_features, num_classes)
    # Input: (B, 3, 224, 224) -> Output: (B, num_classes)
    return model

# Fine-tuning InceptionV3
def get_inception(num_classes=100, pretrained=True):
    model = models.inception_v3(weights="IMAGENET1K_V1" if pretrained else None, aux_logits=True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model




# --- SIAMESE NETWORKS ---

# Simple CNN Siamese Network
class SimpleSiamese(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.proj = nn.Sequential(
            nn.Linear(256, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, embedding_dim),
        )

    def forward_once(self, x):
        x = self.cnn(x)
        x = self.proj(x)
        x = F.normalize(x, p=2, dim=1)
        return x

    def forward(self, x1, x2):
        assert x1 is not None and x2 is not None, "SimpleSiamese requires two inputs"
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        return out1, out2

# Siamese Network with SimpleResNet backbone
class SimpleSiameseResNet(nn.Module):
    def __init__(self, embedding_dim=128, backbone_out=512, dropout=0.2):
        super().__init__()
        self.backbone = SimpleResNet(num_classes=backbone_out)
        self.backbone.fc = nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(backbone_out, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward_once(self, x):
        x = self.backbone(x)           # (B, backbone_out)
        x = self.head(x)               # (B, embedding_dim)
        x = F.normalize(x, p=2, dim=1) # L2-normalize
        return x

    def forward(self, x1, x2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        return out1, out2

from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class SiameseEfficientNetB0(nn.Module):
    def __init__(self, embedding_dim=128, pretrained=True):
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b0(weights=weights)
        # Remove last classifier
        self.feature_extractor = nn.Sequential(
            backbone.features,
            backbone.avgpool,  # (B, 1280, 1, 1)
            nn.Flatten(),      # (B, 1280)
        )
        self.embedding = nn.Linear(1280, embedding_dim)

    def forward_once(self, x):
        x = self.feature_extractor(x)
        x = self.embedding(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        return out1, out2

# -------------------------------
# Utility: parameter counting
# -------------------------------
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# -------------------------------
# Base head with BNNeck + L2 norm
# -------------------------------
class EmbeddingHead(nn.Module):
    """
    BNNeck (optional) + Linear -> embedding; always L2-normalized in output.
    """
    def __init__(self, in_dim, embedding_dim=128, use_bnneck=True, p_drop=0.0):
        super().__init__()
        self.use_bnneck = use_bnneck
        self.bnneck = nn.BatchNorm1d(in_dim) if use_bnneck else nn.Identity()
        self.dropout = nn.Dropout(p_drop) if p_drop > 0 else nn.Identity()
        self.fc = nn.Linear(in_dim, embedding_dim, bias=True)

        # init: BN gamma=1, beta=0; linear init "normal"
        if isinstance(self.bnneck, nn.BatchNorm1d):
            nn.init.constant_(self.bnneck.weight, 1.0)
            nn.init.constant_(self.bnneck.bias, 0.0)
        nn.init.normal_(self.fc.weight, 0, 0.01)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x):
        x = self.bnneck(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)  # L2-normalize
        return x

# =========================================================
# 1) ~10-12M params: SiameseResNet18
#    (backbone ≈11.7M; head small -> totale ≈11.8M)
# =========================================================
class SiameseResNet18(nn.Module):
    def __init__(self, embedding_dim=128, pretrained=True, freeze_bn=False,
                 use_bnneck=True, p_drop=0.0):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        # remove classifier and keep everything up to GAP
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # -> (B, 512, 1, 1)
        self.head = EmbeddingHead(in_dim=512, embedding_dim=embedding_dim,
                                  use_bnneck=use_bnneck, p_drop=p_drop)
        self._freeze_bn = freeze_bn

    def forward_once(self, x):
        # x: (B,3,H,W) normalized ImageNet
        f = self.backbone(x)                  # (B, 512, 1, 1)
        f = torch.flatten(f, 1)               # (B, 512)
        z = self.head(f)                      # (B, embedding_dim) L2-normalized
        return z

    @torch.no_grad()
    def set_batchnorm_eval(self):
        # useful with small batches
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.eval()

    def forward(self, x1, x2):
        if getattr(self, "_freeze_bn", False):
            self.set_batchnorm_eval()
        z1 = self.forward_once(x1)
        z2 = self.forward_once(x2)
        return z1, z2

# =========================================================
# 2) ~30M params: SiameseResNet50_30M
#    (backbone ≈25.6M + head 2048->2048->emb ≈4.45M => ~30.0M)
# =========================================================
class SiameseResNet50_30M(nn.Module):
    def __init__(self, embedding_dim=128, pretrained=True, freeze_bn=False,
                 use_bnneck=True, p_drop=0.0):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # -> (B, 2048, 1, 1)

# Head "large" to reach ~30M:
        # 2048 -> 2048 (Linear + BN + ReLU + Dropout) -> EmbeddingHead
        self.proj = nn.Sequential(
            nn.Linear(2048, 2048, bias=True),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop) if p_drop > 0 else nn.Identity(),
        )
        nn.init.kaiming_normal_(self.proj[0].weight, nonlinearity='relu')
        nn.init.constant_(self.proj[0].bias, 0.0)
        nn.init.constant_(self.proj[1].weight, 1.0)
        nn.init.constant_(self.proj[1].bias, 0.0)

        self.head = EmbeddingHead(in_dim=2048, embedding_dim=embedding_dim,
                                  use_bnneck=use_bnneck, p_drop=0.0)  # dropout already in proj
        self._freeze_bn = freeze_bn

    def forward_once(self, x):
        f = self.backbone(x)            # (B, 2048, 1, 1)
        f = torch.flatten(f, 1)         # (B, 2048)
        f = self.proj(f)                # (B, 2048)
        z = self.head(f)                # (B, embedding_dim) L2-normalized
        return z

    @torch.no_grad()
    def set_batchnorm_eval(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.eval()

    def forward(self, x1, x2):
        if getattr(self, "_freeze_bn", False):
            self.set_batchnorm_eval()
        z1 = self.forward_once(x1)
        z2 = self.forward_once(x2)
        return z1, z2