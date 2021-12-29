"""Clean Weather Model."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 12月 14日 星期二 00:22:28 CST
# ***
# ************************************************************************************/
#
import math
import torch
import torch.nn as nn
import torch.nn.functional
import torch.nn.functional as F
from functools import partial


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvLayer, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # print(self.conv2d)
        # Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # Conv2d(8, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x):
        out = self.conv2d(x)
        return out


class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(UpsampleConvLayer, self).__init__()
        self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=1)
        # self = UpsampleConvLayer(
        # (conv2d): ConvTranspose2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)), xxxx8888
        # )
        # in_channels = 512
        # out_channels = 512
        # kernel_size = 4
        # stride = 2

    def forward(self, x):
        out = self.conv2d(x)
        return out


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1
        out = torch.add(out, residual)
        return out


class CleanWeatherModel(nn.Module):
    def __init__(self):
        super(CleanWeatherModel, self).__init__()
        self.Tenc = Tenc()
        self.Tdec = Tdec()
        self.convtail = convprojection()
        self.clean = ConvLayer(8, 3, kernel_size=3, stride=1, padding=1)
        self.active = nn.Tanh()

    def forward(self, x):
        # x.size() -- torch.Size([1, 3, 368, 640])
        x1 = self.Tenc(x)
        # len(x1), x1[0].size(), x1[1].size(),x1[2].size(), x1[3].size()
        # (4, torch.Size([1, 64, 92, 160]),
        #     torch.Size([1, 128, 46, 80]),
        #     torch.Size([1, 320, 23, 40]),
        #     torch.Size([1, 512, 12, 20]))
        x2 = self.Tdec(x1)
        # len(x2), x2[0].size() -- (1, torch.Size([1, 512, 6, 10]))
        x = self.convtail(x1, x2)
        # x.size() -- torch.Size([1, 8, 368, 640])

        clean = self.active(self.clean(x))
        # self.clean(x).size() -- torch.Size([1, 3, 368, 640])

        return clean.clamp(0.0, 1.0)


class EncoderTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dims=[64, 128, 256, 512],
        num_heads=[1, 2, 4, 8],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        depths=[3, 4, 6, 3],
        sr_ratios=[8, 4, 2, 1],
    ):
        super().__init__()
        # self = Tenc()
        # img_size = 224
        # patch_size = 4
        # in_chans = 3
        # num_classes = 1000
        # embed_dims = [64, 128, 320, 512]
        # num_heads = [1, 2, 4, 4]
        # mlp_ratios = [2, 2, 2, 2]
        # qkv_bias = True
        # qk_scale = None
        # drop_rate = 0.0
        # attn_drop_rate = 0.0
        # drop_path_rate = 0.1
        # norm_layer = functools.partial(<class 'torch.nn.modules.normalization.LayerNorm'>, eps=1e-06)
        # depths = [2, 2, 2, 2]
        # sr_ratios = [4, 2, 2, 1]

        self.num_classes = num_classes
        self.depths = depths

        # patch embedding definitions
        self.patch_embed1 = OverlapPatchEmbed(
            img_size=img_size, patch_size=7, stride=4, in_chans=in_chans, embed_dim=embed_dims[0]
        )
        self.patch_embed2 = OverlapPatchEmbed(
            img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1]
        )
        self.patch_embed3 = OverlapPatchEmbed(
            img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2]
        )
        self.patch_embed4 = OverlapPatchEmbed(
            img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3]
        )

        # (patch_embed1): OverlapPatchEmbed(
        # (proj): Conv2d(3, 64, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3))
        # (norm): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        # )
        # (patch_embed2): OverlapPatchEmbed(
        # (proj): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        # )
        # (patch_embed3): OverlapPatchEmbed(
        # (proj): Conv2d(128, 320, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # (norm): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
        # )
        # (patch_embed4): OverlapPatchEmbed(
        # (proj): Conv2d(320, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # (norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        # )

        # for Intra-patch transformer blocks
        self.mini_patch_embed1 = OverlapPatchEmbed(
            img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1]
        )
        self.mini_patch_embed2 = OverlapPatchEmbed(
            img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2]
        )
        self.mini_patch_embed3 = OverlapPatchEmbed(
            img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3]
        )
        self.mini_patch_embed4 = OverlapPatchEmbed(
            img_size=img_size // 32, patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[3]
        )
        # (mini_patch_embed1): OverlapPatchEmbed(
        # (proj): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        # )
        # (mini_patch_embed2): OverlapPatchEmbed(
        # (proj): Conv2d(128, 320, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # (norm): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
        # )
        # (mini_patch_embed3): OverlapPatchEmbed(
        # (proj): Conv2d(320, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # (norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        # )
        # (mini_patch_embed4): OverlapPatchEmbed(
        # (proj): Conv2d(64, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # (norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        # )

        # main  encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[0],
                    num_heads=num_heads[0],
                    mlp_ratio=mlp_ratios[0],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[0],
                )
                for i in range(depths[0])
            ]
        )
        self.norm1 = norm_layer(embed_dims[0])
        # (block1): ModuleList(
        # (0): Block(
        #   (norm1): LayerNorm((64,), eps=1e-06, elementwise_affine=True)
        #   (attn): Attention(
        #     (q): Linear(in_features=64, out_features=64, bias=True)
        #     (kv): Linear(in_features=64, out_features=128, bias=True)
        #     (attn_drop): Dropout(p=0.0, inplace=False)
        #     (proj): Linear(in_features=64, out_features=64, bias=True)
        #     (proj_drop): Dropout(p=0.0, inplace=False)
        #     (sr): Conv2d(64, 64, kernel_size=(4, 4), stride=(4, 4))
        #     (norm): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        #   )
        #   (drop_path): Identity()
        #   (norm2): LayerNorm((64,), eps=1e-06, elementwise_affine=True)
        #   (mlp): Mlp(
        #     (fc1): Linear(in_features=64, out_features=128, bias=True)
        #     (dwconv): DWConv(
        #       (dwconv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128)
        #     )
        #     (act): GELU()
        #     (fc2): Linear(in_features=128, out_features=64, bias=True)
        #     (drop): Dropout(p=0.0, inplace=False)
        #   )
        # )
        # (1): Block(
        #   (norm1): LayerNorm((64,), eps=1e-06, elementwise_affine=True)
        #   (attn): Attention(
        #     (q): Linear(in_features=64, out_features=64, bias=True)
        #     (kv): Linear(in_features=64, out_features=128, bias=True)
        #     (attn_drop): Dropout(p=0.0, inplace=False)
        #     (proj): Linear(in_features=64, out_features=64, bias=True)
        #     (proj_drop): Dropout(p=0.0, inplace=False)
        #     (sr): Conv2d(64, 64, kernel_size=(4, 4), stride=(4, 4))
        #     (norm): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        #   )
        #   (drop_path): DropPath()
        #   (norm2): LayerNorm((64,), eps=1e-06, elementwise_affine=True)
        #   (mlp): Mlp(
        #     (fc1): Linear(in_features=64, out_features=128, bias=True)
        #     (dwconv): DWConv(
        #       (dwconv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128)
        #     )
        #     (act): GELU()
        #     (fc2): Linear(in_features=128, out_features=64, bias=True)
        #     (drop): Dropout(p=0.0, inplace=False)
        #   )
        # )
        # )
        # (norm1): LayerNorm((64,), eps=1e-06, elementwise_affine=True)

        # intra-patch encoder
        self.patch_block1 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[1],
                    num_heads=num_heads[0],
                    mlp_ratio=mlp_ratios[0],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[0],
                )
                for i in range(1)
            ]
        )
        self.pnorm1 = norm_layer(embed_dims[1])
        # (patch_block1): ModuleList(
        # (0): Block(
        #   (norm1): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
        #   (attn): Attention(
        #     (q): Linear(in_features=128, out_features=128, bias=True)
        #     (kv): Linear(in_features=128, out_features=256, bias=True)
        #     (attn_drop): Dropout(p=0.0, inplace=False)
        #     (proj): Linear(in_features=128, out_features=128, bias=True)
        #     (proj_drop): Dropout(p=0.0, inplace=False)
        #     (sr): Conv2d(128, 128, kernel_size=(4, 4), stride=(4, 4))
        #     (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        #   )
        #   (drop_path): Identity()
        #   (norm2): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
        #   (mlp): Mlp(
        #     (fc1): Linear(in_features=128, out_features=256, bias=True)
        #     (dwconv): DWConv(
        #       (dwconv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256)
        #     )
        #     (act): GELU()
        #     (fc2): Linear(in_features=256, out_features=128, bias=True)
        #     (drop): Dropout(p=0.0, inplace=False)
        #   )
        # )
        # )
        # (pnorm1): LayerNorm((128,), eps=1e-06, elementwise_affine=True)

        # main  encoder
        cur += depths[0]
        self.block2 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[1],
                    num_heads=num_heads[1],
                    mlp_ratio=mlp_ratios[1],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[1],
                )
                for i in range(depths[1])
            ]
        )
        self.norm2 = norm_layer(embed_dims[1])
        # (patch_block2): ModuleList(
        # (0): Block(
        #   (norm1): LayerNorm((320,), eps=1e-06, elementwise_affine=True)
        #   (attn): Attention(
        #     (q): Linear(in_features=320, out_features=320, bias=True)
        #     (kv): Linear(in_features=320, out_features=640, bias=True)
        #     (attn_drop): Dropout(p=0.0, inplace=False)
        #     (proj): Linear(in_features=320, out_features=320, bias=True)
        #     (proj_drop): Dropout(p=0.0, inplace=False)
        #     (sr): Conv2d(320, 320, kernel_size=(2, 2), stride=(2, 2))
        #     (norm): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
        #   )
        #   (drop_path): DropPath()
        #   (norm2): LayerNorm((320,), eps=1e-06, elementwise_affine=True)
        #   (mlp): Mlp(
        #     (fc1): Linear(in_features=320, out_features=640, bias=True)
        #     (dwconv): DWConv(
        #       (dwconv): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=640)
        #     )
        #     (act): GELU()
        #     (fc2): Linear(in_features=640, out_features=320, bias=True)
        #     (drop): Dropout(p=0.0, inplace=False)
        #   )
        # )
        # )
        # (pnorm2): LayerNorm((320,), eps=1e-06, elementwise_affine=True)

        # intra-patch encoder
        self.patch_block2 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[2],
                    num_heads=num_heads[1],
                    mlp_ratio=mlp_ratios[1],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[1],
                )
                for i in range(1)
            ]
        )
        self.pnorm2 = norm_layer(embed_dims[2])
        # (patch_block2): ModuleList(
        # (0): Block(
        #   (norm1): LayerNorm((320,), eps=1e-06, elementwise_affine=True)
        #   (attn): Attention(
        #     (q): Linear(in_features=320, out_features=320, bias=True)
        #     (kv): Linear(in_features=320, out_features=640, bias=True)
        #     (attn_drop): Dropout(p=0.0, inplace=False)
        #     (proj): Linear(in_features=320, out_features=320, bias=True)
        #     (proj_drop): Dropout(p=0.0, inplace=False)
        #     (sr): Conv2d(320, 320, kernel_size=(2, 2), stride=(2, 2))
        #     (norm): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
        #   )
        #   (drop_path): DropPath()
        #   (norm2): LayerNorm((320,), eps=1e-06, elementwise_affine=True)
        #   (mlp): Mlp(
        #     (fc1): Linear(in_features=320, out_features=640, bias=True)
        #     (dwconv): DWConv(
        #       (dwconv): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=640)
        #     )
        #     (act): GELU()
        #     (fc2): Linear(in_features=640, out_features=320, bias=True)
        #     (drop): Dropout(p=0.0, inplace=False)
        #   )
        # )
        # )
        # (pnorm2): LayerNorm((320,), eps=1e-06, elementwise_affine=True)

        # main  encoder
        cur += depths[1]
        self.block3 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[2],
                    num_heads=num_heads[2],
                    mlp_ratio=mlp_ratios[2],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[2],
                )
                for i in range(depths[2])
            ]
        )
        self.norm3 = norm_layer(embed_dims[2])
        # (block3): ModuleList(
        # (0): Block(
        #   (norm1): LayerNorm((320,), eps=1e-06, elementwise_affine=True)
        #   (attn): Attention(
        #     (q): Linear(in_features=320, out_features=320, bias=True)
        #     (kv): Linear(in_features=320, out_features=640, bias=True)
        #     (attn_drop): Dropout(p=0.0, inplace=False)
        #     (proj): Linear(in_features=320, out_features=320, bias=True)
        #     (proj_drop): Dropout(p=0.0, inplace=False)
        #     (sr): Conv2d(320, 320, kernel_size=(2, 2), stride=(2, 2))
        #     (norm): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
        #   )
        #   (drop_path): DropPath()
        #   (norm2): LayerNorm((320,), eps=1e-06, elementwise_affine=True)
        #   (mlp): Mlp(
        #     (fc1): Linear(in_features=320, out_features=640, bias=True)
        #     (dwconv): DWConv(
        #       (dwconv): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=640)
        #     )
        #     (act): GELU()
        #     (fc2): Linear(in_features=640, out_features=320, bias=True)
        #     (drop): Dropout(p=0.0, inplace=False)
        #   )
        # )
        # (1): Block(
        #   (norm1): LayerNorm((320,), eps=1e-06, elementwise_affine=True)
        #   (attn): Attention(
        #     (q): Linear(in_features=320, out_features=320, bias=True)
        #     (kv): Linear(in_features=320, out_features=640, bias=True)
        #     (attn_drop): Dropout(p=0.0, inplace=False)
        #     (proj): Linear(in_features=320, out_features=320, bias=True)
        #     (proj_drop): Dropout(p=0.0, inplace=False)
        #     (sr): Conv2d(320, 320, kernel_size=(2, 2), stride=(2, 2))
        #     (norm): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
        #   )
        #   (drop_path): DropPath()
        #   (norm2): LayerNorm((320,), eps=1e-06, elementwise_affine=True)
        #   (mlp): Mlp(
        #     (fc1): Linear(in_features=320, out_features=640, bias=True)
        #     (dwconv): DWConv(
        #       (dwconv): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=640)
        #     )
        #     (act): GELU()
        #     (fc2): Linear(in_features=640, out_features=320, bias=True)
        #     (drop): Dropout(p=0.0, inplace=False)
        #   )
        # )
        # )
        # (norm3): LayerNorm((320,), eps=1e-06, elementwise_affine=True)

        # intra-patch encoder
        self.patch_block3 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[3],
                    num_heads=num_heads[1],
                    mlp_ratio=mlp_ratios[2],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[2],
                )
                for i in range(1)
            ]
        )
        self.pnorm3 = norm_layer(embed_dims[3])
        # (patch_block3): ModuleList(
        # (0): Block(
        #   (norm1): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        #   (attn): Attention(
        #     (q): Linear(in_features=512, out_features=512, bias=True)
        #     (kv): Linear(in_features=512, out_features=1024, bias=True)
        #     (attn_drop): Dropout(p=0.0, inplace=False)
        #     (proj): Linear(in_features=512, out_features=512, bias=True)
        #     (proj_drop): Dropout(p=0.0, inplace=False)
        #     (sr): Conv2d(512, 512, kernel_size=(2, 2), stride=(2, 2))
        #     (norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        #   )
        #   (drop_path): DropPath()
        #   (norm2): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        #   (mlp): Mlp(
        #     (fc1): Linear(in_features=512, out_features=1024, bias=True)
        #     (dwconv): DWConv(
        #       (dwconv): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1024)
        #     )
        #     (act): GELU()
        #     (fc2): Linear(in_features=1024, out_features=512, bias=True)
        #     (drop): Dropout(p=0.0, inplace=False)
        #   )
        # )
        # )
        # (pnorm3): LayerNorm((512,), eps=1e-06, elementwise_affine=True)

        # main  encoder
        cur += depths[2]
        self.block4 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[3],
                    num_heads=num_heads[3],
                    mlp_ratio=mlp_ratios[3],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[3],
                )
                for i in range(depths[3])
            ]
        )
        self.norm4 = norm_layer(embed_dims[3])
        # (block4): ModuleList(
        # (0): Block(
        #   (norm1): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        #   (attn): Attention(
        #     (q): Linear(in_features=512, out_features=512, bias=True)
        #     (kv): Linear(in_features=512, out_features=1024, bias=True)
        #     (attn_drop): Dropout(p=0.0, inplace=False)
        #     (proj): Linear(in_features=512, out_features=512, bias=True)
        #     (proj_drop): Dropout(p=0.0, inplace=False)
        #   )
        #   (drop_path): DropPath()
        #   (norm2): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        #   (mlp): Mlp(
        #     (fc1): Linear(in_features=512, out_features=1024, bias=True)
        #     (dwconv): DWConv(
        #       (dwconv): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1024)
        #     )
        #     (act): GELU()
        #     (fc2): Linear(in_features=1024, out_features=512, bias=True)
        #     (drop): Dropout(p=0.0, inplace=False)
        #   )
        # )
        # (1): Block(
        #   (norm1): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        #   (attn): Attention(
        #     (q): Linear(in_features=512, out_features=512, bias=True)
        #     (kv): Linear(in_features=512, out_features=1024, bias=True)
        #     (attn_drop): Dropout(p=0.0, inplace=False)
        #     (proj): Linear(in_features=512, out_features=512, bias=True)
        #     (proj_drop): Dropout(p=0.0, inplace=False)
        #   )
        #   (drop_path): DropPath()
        #   (norm2): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        #   (mlp): Mlp(
        #     (fc1): Linear(in_features=512, out_features=1024, bias=True)
        #     (dwconv): DWConv(
        #       (dwconv): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1024)
        #     )
        #     (act): GELU()
        #     (fc2): Linear(in_features=1024, out_features=512, bias=True)
        #     (drop): Dropout(p=0.0, inplace=False)
        #   )
        # )
        # )
        # (norm4): LayerNorm((512,), eps=1e-06, elementwise_affine=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        # pp x.size() -- torch.Size([1, 3, 368, 640])
        B = x.shape[0]
        outs = []
        embed_dims = [64, 128, 320, 512]
        # stage 1
        x1, H1, W1 = self.patch_embed1(x)
        x2, H2, W2 = self.mini_patch_embed1(x1.permute(0, 2, 1).reshape(B, embed_dims[0], H1, W1))

        for i, blk in enumerate(self.block1):
            x1 = blk(x1, H1, W1)
        x1 = self.norm1(x1)
        x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()

        for i, blk in enumerate(self.patch_block1):
            x2 = blk(x2, H2, W2)
        x2 = self.pnorm1(x2)
        x2 = x2.reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()

        outs.append(x1)

        # stage 2
        x1, H1, W1 = self.patch_embed2(x1)
        x1 = x1.permute(0, 2, 1).reshape(B, embed_dims[1], H1, W1) + x2
        x2, H2, W2 = self.mini_patch_embed2(x1)

        x1 = x1.view(x1.shape[0], x1.shape[1], -1).permute(0, 2, 1)

        for i, blk in enumerate(self.block2):
            x1 = blk(x1, H1, W1)
        x1 = self.norm2(x1)
        x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x1)

        for i, blk in enumerate(self.patch_block2):
            x2 = blk(x2, H2, W2)
        x2 = self.pnorm2(x2)
        x2 = x2.reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()

        # stage 3
        x1, H1, W1 = self.patch_embed3(x1)
        x1 = x1.permute(0, 2, 1).reshape(B, embed_dims[2], H1, W1) + x2
        x2, H2, W2 = self.mini_patch_embed3(x1)

        x1 = x1.view(x1.shape[0], x1.shape[1], -1).permute(0, 2, 1)

        for i, blk in enumerate(self.block3):
            x1 = blk(x1, H1, W1)
        x1 = self.norm3(x1)
        x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x1)

        for i, blk in enumerate(self.patch_block3):
            x2 = blk(x2, H2, W2)
        x2 = self.pnorm3(x2)
        x2 = x2.reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()

        # stage 4
        x1, H1, W1 = self.patch_embed4(x1)
        x1 = x1.permute(0, 2, 1).reshape(B, embed_dims[3], H1, W1) + x2

        x1 = x1.view(x1.shape[0], x1.shape[1], -1).permute(0, 2, 1)

        for i, blk in enumerate(self.block4):
            x1 = blk(x1, H1, W1)
        x1 = self.norm4(x1)
        x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x1)

        # len(outs), outs[0].size(), outs[1].size(), outs[2].size(), outs[3].size()
        # (4, torch.Size([1, 64, 92, 160]),
        #     torch.Size([1, 128, 46, 80]),
        #     torch.Size([1, 320, 23, 40]),
        #     torch.Size([1, 512, 12, 20]))

        return outs


class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=(patch_size[0] // 2, patch_size[1] // 2)
        )
        self.norm = nn.LayerNorm(embed_dim)

        # self = OverlapPatchEmbed(
        #   (proj): Conv2d(3, 64, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3))
        #   (norm): LayerNorm((64,), eps=1e-05, elementwise_affine=True), xxxx8888
        # )
        # img_size = (224, 224)
        # patch_size = (7, 7)
        # stride = 4
        # in_chans = 3
        # embed_dim = 64

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        # x.size() -- torch.Size([1, 3, 368, 640])
        x = self.proj(x)
        # x.size() -- torch.Size([1, 64, 92, 160])
        _, _, H, W = x.shape
        # 92*160 -- 14720
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        # x.size() -- torch.Size([1, 14720, 64])

        return x, H, W


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)
        # self = Mlp(
        #   (fc1): Linear(in_features=64, out_features=128, bias=True)
        #   (dwconv): DWConv(
        #     (dwconv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128)
        #   )
        #   (act): GELU()
        #   (fc2): Linear(in_features=128, out_features=64, bias=True)
        #   (drop): Dropout(p=0.0, inplace=False)
        # )
        # in_features = 64
        # hidden_features = 128
        # out_features = 64
        # act_layer = <class 'torch.nn.modules.activation.GELU'>
        # drop = 0.0

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)
        # self = Attention(
        #   (q): Linear(in_features=64, out_features=64, bias=True)
        #   (kv): Linear(in_features=64, out_features=128, bias=True)
        #   (attn_drop): Dropout(p=0.0, inplace=False)
        #   (proj): Linear(in_features=64, out_features=64, bias=True)
        #   (proj_drop): Dropout(p=0.0, inplace=False)
        #   (sr): Conv2d(64, 64, kernel_size=(4, 4), stride=(4, 4))
        #   (norm): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        # )
        # dim = 64
        # num_heads = 1
        # qkv_bias = True
        # qk_scale = None
        # attn_drop = 0.0
        # proj_drop = 0.0
        # sr_ratio = 4

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):

        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Attention_dec(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        # print("qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1 ?")
        # print(qkv_bias, qk_scale, attn_drop, proj_drop, sr_ratio)
        # True None 0.0 0.0 1
        # True None 0.0 0.0 1
        # True None 0.0 0.0 1

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.task_query = nn.Parameter(torch.randn(1, 48, dim))
        self.sr_ratio = sr_ratio

        # sr_ratio == 1
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        # xxxx8888 onnx ?
        # https://docs.marklogic.com/guide/app-dev/PyTorch
        # print("x.size(): ", x.size())
        # x.size():  torch.Size([1, 60, 512])
        # x.size():  torch.Size([1, 96, 512])

        B, N, C = x.shape
        task_q = self.task_query

        # This is because we fix the task parameters to be of a certain dimension, so with varying batch size, we just stack up the same queries to operate on the entire batch
        if B > 1:

            task_q = task_q.unsqueeze(0).repeat(B, 1, 1, 1)
            task_q = task_q.squeeze(1)

        q = self.q(task_q).reshape(B, task_q.shape[1], self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        q = torch.nn.functional.interpolate(q, size=(v.shape[2], v.shape[3]))
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block_dec(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_dec(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )

        # drop_path: ------- 0.0
        # drop_path: ------- 0.014285714365541935
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):

        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        # torch._shape_as_tensor(x)
        # torch._reshape_from_tensor(x, shape)
        B, N, C = x.shape
        H_W_shape = torch.IntTensor([H, W])

        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class DecoderTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dims=[64, 128, 256, 512],
        num_heads=[1, 2, 4, 8],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        depths=[3, 4, 6, 3],
        sr_ratios=[8, 4, 2, 1],
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(
            img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[3], embed_dim=embed_dims[3]
        )

        # transformer decoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList(
            [
                Block_dec(
                    dim=embed_dims[3],
                    num_heads=num_heads[3],
                    mlp_ratio=mlp_ratios[3],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[3],
                )
                for i in range(depths[0])
            ]
        )
        self.norm1 = norm_layer(embed_dims[3])

        # img_size = 224
        # patch_size = 4
        # in_chans = 3
        # num_classes = 1000
        # embed_dims = [64, 128, 320, 512]
        # num_heads = [1, 2, 5, 8]
        # mlp_ratios = [4, 4, 4, 4]
        # qkv_bias = True
        # qk_scale = None
        # drop_rate = 0.0
        # attn_drop_rate = 0.0
        # drop_path_rate = 0.1
        # norm_layer = functools.partial(<class 'torch.nn.modules.normalization.LayerNorm'>, eps=1e-06)
        # depths = [3, 4, 6, 3]
        # sr_ratios = [8, 4, 2, 1]
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        # len(x), x[0].size(), x[1].size(), x[2].size(), x[3].size()
        # (4,
        #  torch.Size([1, 64, 92, 160]),
        #  torch.Size([1, 128, 46, 80]),
        #  torch.Size([1, 320, 23, 40]),
        #  torch.Size([1, 512, 12, 20]))

        x = x[3]
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        # (Pdb) len(outs), outs[0].size()
        # (1, torch.Size([1, 512, 6, 10]))
        return outs


class Tenc(EncoderTransformer):
    def __init__(self):
        super(Tenc, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 4, 4],
            mlp_ratios=[2, 2, 2, 2],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[2, 2, 2, 2],
            sr_ratios=[4, 2, 2, 1],
            drop_rate=0.0,
            drop_path_rate=0.1,
        )


class Tdec(DecoderTransformer):
    def __init__(self):
        super(Tdec, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 4, 6, 3],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0,
            drop_path_rate=0.1,
        )


class convprojection(nn.Module):
    def __init__(self):
        super(convprojection, self).__init__()

        self.convd32x = UpsampleConvLayer(512, 512, kernel_size=4, stride=2)
        self.convd16x = UpsampleConvLayer(512, 320, kernel_size=4, stride=2)
        self.dense_4 = nn.Sequential(ResidualBlock(320))
        self.convd8x = UpsampleConvLayer(320, 128, kernel_size=4, stride=2)
        self.dense_3 = nn.Sequential(ResidualBlock(128))
        self.convd4x = UpsampleConvLayer(128, 64, kernel_size=4, stride=2)
        self.dense_2 = nn.Sequential(ResidualBlock(64))
        self.convd2x = UpsampleConvLayer(64, 16, kernel_size=4, stride=2)
        self.dense_1 = nn.Sequential(ResidualBlock(16))
        self.convd1x = UpsampleConvLayer(16, 8, kernel_size=4, stride=2)
        self.conv_output = ConvLayer(8, 3, kernel_size=3, stride=1, padding=1)

        self.active = nn.Tanh()
        # self = convprojection(
        #   (convd32x): UpsampleConvLayer(
        #     (conv2d): ConvTranspose2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        #   )
        #   (convd16x): UpsampleConvLayer(
        #     (conv2d): ConvTranspose2d(512, 320, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        #   )
        #   (dense_4): Sequential(
        #     (0): ResidualBlock(
        #       (conv1): ConvLayer(
        #         (conv2d): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       )
        #       (conv2): ConvLayer(
        #         (conv2d): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       )
        #       (relu): ReLU()
        #     )
        #   )
        #   (convd8x): UpsampleConvLayer(
        #     (conv2d): ConvTranspose2d(320, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        #   )
        #   (dense_3): Sequential(
        #     (0): ResidualBlock(
        #       (conv1): ConvLayer(
        #         (conv2d): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       )
        #       (conv2): ConvLayer(
        #         (conv2d): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       )
        #       (relu): ReLU()
        #     )
        #   )
        #   (convd4x): UpsampleConvLayer(
        #     (conv2d): ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        #   )
        #   (dense_2): Sequential(
        #     (0): ResidualBlock(
        #       (conv1): ConvLayer(
        #         (conv2d): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       )
        #       (conv2): ConvLayer(
        #         (conv2d): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       )
        #       (relu): ReLU()
        #     )
        #   )
        #   (convd2x): UpsampleConvLayer(
        #     (conv2d): ConvTranspose2d(64, 16, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        #   )
        #   (dense_1): Sequential(
        #     (0): ResidualBlock(
        #       (conv1): ConvLayer(
        #         (conv2d): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       )
        #       (conv2): ConvLayer(
        #         (conv2d): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       )
        #       (relu): ReLU()
        #     )
        #   )
        #   (convd1x): UpsampleConvLayer(
        #     (conv2d): ConvTranspose2d(16, 8, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        #   )
        #   (conv_output): ConvLayer(
        #     (conv2d): Conv2d(8, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #   )
        #   (active): Tanh()
        # )

    def forward(self, x1, x2):
        # len(x1), x1[0].size(), x1[1].size(), x1[2].size(), x1[3].size()
        # (4, torch.Size([1, 64, 92, 160]),
        #     torch.Size([1, 128, 46, 80]),
        #     torch.Size([1, 320, 23, 40]),
        #     torch.Size([1, 512, 12, 20]))
        # len(x2),x2[0].size() -- (1, torch.Size([1, 512, 6, 10]))

        res32x = self.convd32x(x2[0])

        if x1[3].shape[3] != res32x.shape[3] and x1[3].shape[2] != res32x.shape[2]:
            p2d = (0, -1, 0, -1)
            res32x = F.pad(res32x, p2d, "constant", 0)

        elif x1[3].shape[3] != res32x.shape[3] and x1[3].shape[2] == res32x.shape[2]:
            p2d = (0, -1, 0, 0)
            res32x = F.pad(res32x, p2d, "constant", 0)
        elif x1[3].shape[3] == res32x.shape[3] and x1[3].shape[2] != res32x.shape[2]:
            p2d = (0, 0, 0, -1)
            res32x = F.pad(res32x, p2d, "constant", 0)

        res16x = res32x + x1[3]
        res16x = self.convd16x(res16x)

        if x1[2].shape[3] != res16x.shape[3] and x1[2].shape[2] != res16x.shape[2]:
            p2d = (0, -1, 0, -1)
            res16x = F.pad(res16x, p2d, "constant", 0)
        elif x1[2].shape[3] != res16x.shape[3] and x1[2].shape[2] == res16x.shape[2]:
            p2d = (0, -1, 0, 0)
            res16x = F.pad(res16x, p2d, "constant", 0)
        elif x1[2].shape[3] == res16x.shape[3] and x1[2].shape[2] != res16x.shape[2]:
            p2d = (0, 0, 0, -1)
            res16x = F.pad(res16x, p2d, "constant", 0)

        res8x = self.dense_4(res16x) + x1[2]
        res8x = self.convd8x(res8x)
        res4x = self.dense_3(res8x) + x1[1]
        res4x = self.convd4x(res4x)
        res2x = self.dense_2(res4x) + x1[0]
        res2x = self.convd2x(res2x)
        x = res2x
        x = self.dense_1(x)
        x = self.convd1x(x)
        # x.size() -- torch.Size([1, 8, 368, 640])

        return x


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


if __name__ == "__main__":
    model = CleanWeatherModel()
    print(model)
