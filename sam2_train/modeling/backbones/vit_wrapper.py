import torch.nn as nn
from .vit import ViT  # 原有的 ViT
from .vitcomer import ViTCoMer  # 新的 ViTCoMer

class ViTWrapper(nn.Module):
    def __init__(self, use_vitcomer=False, vit_kwargs=None, vitcomer_kwargs=None):
        super(ViTWrapper, self).__init__()
        
        # 選擇架構
        if use_vitcomer:
            self.model = ViTCoMer(**(vitcomer_kwargs or {}))
        else:
            self.model = ViT(**(vit_kwargs or {}))

    def forward(self, x):
        return self.model(x)
