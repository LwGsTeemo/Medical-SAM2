import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block

class ViTCoMer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        embed_dims=[768, 384, 192, 96],
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        return_interm_layers=False,
        stages=[2, 3, 16, 3],  
        global_att_blocks=[12, 16], 
        window_spec=[8, 4, 14, 7],
        dim_mul=2.0,
        head_mul=2.0,
        min_head_dim=1,  
        min_embed_dim=32,
    ):
        super(ViTCoMer, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dims = embed_dims
        self.depth = depth
        self.return_interm_layers = return_interm_layers
        self.stages = stages
        self.global_att_blocks = global_att_blocks
        self.window_spec = window_spec
        self.dim_mul = dim_mul
        self.head_mul = head_mul
        self.min_head_dim = min_head_dim  
        self.min_embed_dim = min_embed_dim

        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=embed_dims[0],  #First stage embed_dim
            kernel_size=patch_size,
            stride=patch_size
        )
        num_patches = (img_size // patch_size) ** 2

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dims[0], num_patches))

        # Transformer blocks with multiple stages
        self.blocks = nn.ModuleList()
        stage_idx = 0
        for i in range(depth):
            if i == sum(stages[:stage_idx + 1]):
                stage_idx += 1
            block_dim = embed_dims[stage_idx]
            num_heads = int(num_heads * (head_mul ** stage_idx))

            # Ensure num_heads and embed_dim are not below the minimum threshold
            head_dim = max(block_dim // num_heads, self.min_head_dim)  # Ensure head_dim > 0
            num_heads = max(block_dim // head_dim, self.min_head_dim)  # Ensure num_heads > 0
            block_dim = max(block_dim, self.min_embed_dim)  # Ensure embed_dim is not too small

            block = Block(  # This should be a custom block like MultiScaleBlock
                dim=block_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate,
            )
            self.blocks.append(block)

        self.norm = nn.LayerNorm(embed_dims[-1])

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, "Input image size must match model size."

        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed

        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if (i + 1) in self.stages:
                outputs.append(x)

        x = self.norm(x)
        return outputs if self.return_interm_layers else x
