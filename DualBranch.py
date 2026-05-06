from torch import nn
from model.resnet50 import *

class DualResNet(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.2):
        super(DualResNet, self).__init__()

        # 保持原始分支结构不变
        self.resnet50_rgb = resnet50(include_top=False, pretrained_path=None)  # 2048, 7, 7  输入通道3
        self.resnet50_noise = resnet50_1(include_top=False, pretrained_path=None)  # 2048, 7, 7 输入通道48 可变


        self.hermite = hermiteConv2D()
        # 高效特征压缩模块
        self.feature_adapter = nn.Sequential(
            nn.Conv2d(4096, 1024, kernel_size=1),  # 1x1卷积降维
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # 自适应池化替代展平
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, num_classes)
        )

    def forward(self, img):
        # 保持原始特征提取
        RGB_feat = self.resnet50_rgb(img)  # [bs, 2048, 7, 7]
        noise_feat = self.hermite(img)
        noise_feat = self.resnet50_noise(noise_feat)


        # 特征拼接
        combined_feat = torch.cat((RGB_feat, noise_feat), dim=1)  # [bs, 4096, 7, 7]

        # 特征压缩
        compressed_feat = self.feature_adapter(combined_feat)  # [bs, 1024, 1, 1]
        compressed_feat = compressed_feat.view(compressed_feat.size(0), -1)  # [bs, 1024]

        # 分类输出
        return self.classifier(compressed_feat)
