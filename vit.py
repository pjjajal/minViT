from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models import named_apply
from timm.models.layers import DropPath
from timm.layers import resample_patch_embed, resample_abs_pos_embed, use_fused_attn


# this is the standard timm vit initialization. 
# for any custom implementation, you can just create a new function with the same signature
# and pass it to the named_apply function in the model constructor.
def init_weights_vit_timm(module: nn.Module, name: str = "") -> None:
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, "init_weights"):
        module.init_weights()

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.default_patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.num_patches = int((img_size // patch_size) ** 2)
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x, patch_size=None):
        if patch_size is None or self.default_patch_size == patch_size:
            x = self.proj(x)
        else:
            new_weight = resample_patch_embed(
                self.proj.weight, (patch_size, patch_size)
            )
            x = F.conv2d(x, weight=new_weight, bias=self.proj.bias, stride=patch_size)
        return x.permute(0, 2, 3, 1).contiguous()


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        out_features,
        act_layer=nn.GELU,
        dropout=0.0,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dropout_1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout_1(x)
        x = self.fc2(x)
        x = self.dropout_2(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.efficient = use_fused_attn()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.attn_drop_prob = attn_drop
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(dim=0)
        if self.efficient:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                scale=self.scale,
                dropout_p=self.attn_drop_prob if self.training else 0.0,
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        path_drop=0.0,
        efficient=False,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        mlp_layer=Mlp,
    ) -> None:
        super().__init__()
        self.attn = Attention(
            embed_dim,
            num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            efficient=efficient,
        )
        self.mlp = mlp_layer(
            in_features=embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            out_features=embed_dim,
            act_layer=act_layer,
            dropout=path_drop,
        )

        self.norm1 = norm_layer(embed_dim)
        self.norm2 = norm_layer(embed_dim)

        self.drop_path1 = DropPath(path_drop) if path_drop > 0.0 else nn.Identity()
        self.drop_path2 = DropPath(path_drop) if path_drop > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        in_chans=3,
        num_classes=1000,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        cls_token=False,
        attn_drop_rate=0.0,
        proj_drop_rate=0.0,
        path_drop_rate=0.0,
        pos_drop_rate=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        mlp_layer=Mlp,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.use_cls_token = cls_token  # whether to include cls token
        self.prefix_tokens = 1 if cls_token else 0
        self.cls_token = (
            nn.Parameter(torch.zeros(1, 1, embed_dim)) if cls_token else None
        )
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.num_tokens = self.patch_embed.num_patches + self.prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_tokens, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(pos_drop_rate)

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    attn_drop=attn_drop_rate,
                    proj_drop=proj_drop_rate,
                    path_drop=path_drop_rate,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    mlp_layer=mlp_layer,
                )
                for _ in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def initialize_weights(self):
        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        if self.reg_token is not None:
            nn.init.normal_(self.reg_token, std=1e-6)

        # we use the timm named_apply to apply the init.
        named_apply(init_weights_vit_timm, self)

    def _pos_embed(self, x, pos_embed):
        B, H, W, C = x.shape
        pos_embed = resample_abs_pos_embed(
            pos_embed, (H, W), num_prefix_tokens=self.prefix_tokens
        )
        x = x.view(B, -1, C)
        x = x + pos_embed
        return self.pos_drop(x)

    def forward_features(self, x, patch_size=None):
        x = self.patch_embed(x, patch_size=patch_size)
        x = self._pos_embed(x, self.pos_embed)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x

    def forward_head(self, x):
        # cls token if present, else avg-pooling
        if self.cls_token:
            x = x[:, 0]
        else:
            x = x.mean(dim=1)
        x = self.head(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def vit_tiny_patch16_224(cls_token=True, **kwargs):
    model_args = dict(
        img_size=224,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        cls_token=cls_token,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    model_args.update(kwargs)
    return VisionTransformer(**model_args)


def vit_small_patch16_224(cls_token=True, **kwargs):
    model_args = dict(
        img_size=224,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        cls_token=cls_token,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    model_args.update(kwargs)
    return VisionTransformer(**model_args)


def vit_base_patch16_224(cls_token=True, **kwargs):
    model_args = dict(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        cls_token=cls_token,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    model_args.update(kwargs)
    return VisionTransformer(**model_args)


def vit_large_patch16_224(cls_token=True, **kwargs):
    model_args = dict(
        img_size=224,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        cls_token=cls_token,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    model_args.update(kwargs)
    return VisionTransformer(**model_args)
