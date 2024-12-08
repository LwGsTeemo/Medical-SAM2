# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
# from .vitcomer import ViTCoMer
# from external.ViTCoMer.segmentation.mmseg_custom.models.backbones.vit_comer import ViTCoMer
# Medical-SAM2/external/ViTCoMer/segmentation/mmseg_custom/models/backbones/vit_comer.py
# from .vitcomer import WrappedViTCoMer  # 或者導入 ViTCoMer

class ImageEncoder(nn.Module):
    def __init__(
        self,
        trunk: nn.Module,
        neck: nn.Module,
        scalp: int = 0,
    ):
        super().__init__()
        self.trunk = trunk
        self.neck = neck
        self.scalp = scalp
        assert (
            self.trunk.embed_dims == self.neck.backbone_channel_list
        ), f"Channel dims of trunk and neck do not match. Trunk: {self.trunk.channel_list}, neck: {self.neck.backbone_channel_list}"

    def forward(self, sample: torch.Tensor):
        # Forward through backbone
        xs = self.trunk(sample)  # trunk 的輸出是 list
        print(f"Trunk output type: {type(xs)}, length: {len(xs)}")
        # 確保每個特徵圖的通道數正確
        adjusted_xs = []
        for i, x in enumerate(xs):
            # print(f"Feature {i} shape before adjustment: {x.shape}")
            x = x.permute(0, 2, 1).unsqueeze(1)  # 調整為 [1, 1, 768, 400]
            print(f"Feature {i} shape after adjustment: {x.shape}")
            adjusted_xs.append(x)
        print(f"Adjusted trunk output shapes: {[x.shape for x in adjusted_xs]}")

        features, pos = self.neck(adjusted_xs)
        if self.scalp > 0:
            # Discard the lowest resolution features
            features, pos = features[: -self.scalp], pos[: -self.scalp]

        src = features[-1]
        output = {
            "vision_features": src,
            "vision_pos_enc": pos,
            "backbone_fpn": features,
        }
        return output


class FpnNeck(nn.Module):
    """
    A modified variant of Feature Pyramid Network (FPN) neck
    (we remove output conv and also do bicubic interpolation similar to ViT
    pos embed interpolation)
    """

    def __init__(
        self,
        position_encoding: nn.Module,
        d_model: int,
        backbone_channel_list: List[int],
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        fpn_interp_model: str = "bilinear",
        fuse_type: str = "sum",
        fpn_top_down_levels: Optional[List[int]] = None,
    ):
        """Initialize the neck
        :param trunk: the backbone
        :param position_encoding: the positional encoding to use
        :param d_model: the dimension of the model
        :param neck_norm: the normalization to use
        """
        super().__init__()
        self.position_encoding = position_encoding
        self.convs = nn.ModuleList()
        self.backbone_channel_list = backbone_channel_list
        for dim in backbone_channel_list:
            current = nn.Sequential()
            current.add_module(
                "conv",
                nn.Conv2d(
                    in_channels=dim,
                    out_channels=d_model,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
            )

            self.convs.append(current)
        self.fpn_interp_model = fpn_interp_model
        assert fuse_type in ["sum", "avg"]
        self.fuse_type = fuse_type

        # levels to have top-down features in its outputs
        # e.g. if fpn_top_down_levels is [2, 3], then only outputs of level 2 and 3
        # have top-down propagation, while outputs of level 0 and level 1 have only
        # lateral features from the same backbone level.
        if fpn_top_down_levels is None:
            # default is to have top-down features on all levels
            fpn_top_down_levels = range(len(self.convs))
        self.fpn_top_down_levels = list(fpn_top_down_levels)

    def forward(self, xs: List[torch.Tensor]):

        out = [None] * len(self.convs)
        pos = [None] * len(self.convs)
        # print(f"len(xs): {len(xs)}, len(self.convs): {len(self.convs)}")
        # print(f"xs: {xs}")  # 打印 xs 的具體內容
        # print(f"self.convs: {self.convs}")  # 打印 conv 層
        # print(f"len(xs): {len(xs)}, len(self.convs): {len(self.convs)}")  # 打印它們的長度

        assert len(xs) == len(self.convs)
        # fpn forward pass
        # see https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/fpn.py
        prev_features = None
        # forward in top-down order (from low to high resolution)
        n = len(self.convs) - 1

        xs = xs[::-1]  # 這將顛倒列表順序

        for i in range(n, -1, -1):
            x = xs[i]
            print(f'before : {x.shape}')
            x = x.permute(3,2,1,0)
            # # 動態調整通道數
            # in_channels = x.shape[1]  # 輸入通道數
            # out_channels = 768  # 輸出通道數
            # if in_channels != out_channels:
            #     conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1).to('cuda:0')
            #     conv.bias.data = conv.bias.data.to('cuda:0').to(torch.bfloat16)
            #     x = conv(x)  # 進行卷積
            # x = x.permute(3, 1, 2, 0)
            print(f'after : {x.shape}')
            lateral_features = self.convs[n - i](x)
            print(f'shape : {lateral_features.shape}')
            if i in self.fpn_top_down_levels and prev_features is not None:
                top_down_features = F.interpolate(
                    prev_features.to(dtype=torch.float32),
                    scale_factor=2.0,
                    mode=self.fpn_interp_model,
                    align_corners=(
                        None if self.fpn_interp_model == "nearest" else False
                    ),
                    antialias=False,
                )
                prev_features = lateral_features + top_down_features
                if self.fuse_type == "avg":
                    prev_features /= 2
            else:
                prev_features = lateral_features
            x_out = prev_features
            out[i] = x_out
            pos[i] = self.position_encoding(x_out).to(x_out.dtype)

        return out, pos
