from .vitcomer import ViTCoMer  # 引入 ViTCoMer 類別

import torch.nn as nn

class WrappedViTCoMer(nn.Module):
    def __init__(self, vitcomer, backbone_channel_list):
        super().__init__()
        self.vitcomer = vitcomer
        self.backbone_channel_list = backbone_channel_list

    def forward(self, x):
        # 獲取 ViTCoMer 的多層輸出
        features = self.vitcomer(x)  # [f1, f2, f3, f4]
        # 檢查通道數是否符合 backbone_channel_list
        assert len(features) == len(self.backbone_channel_list), \
            f"Expected {len(self.backbone_channel_list)} features, got {len(features)}"
        for i, (feat, expected_dim) in enumerate(zip(features, self.backbone_channel_list)):
            assert feat.size(1) == expected_dim, \
                f"Feature {i} channel mismatch: expected {expected_dim}, got {feat.size(1)}"
        return features
