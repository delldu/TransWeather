import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from typing import Tuple
from typing import Dict

import pdb


model_config = {
    "xxs": {
        "features": [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320],
        "dims": [64, 80, 96],
        "layers": [2, 4, 3],
        "expansion_ratio": 2,
    },
    "xs": {
        "features": [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384],
        "dims": [96, 120, 144],
        "layers": [2, 4, 3],
        "expansion_ratio": 4,
    },
    "s": {
        "features": [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640],
        "dims": [144, 192, 240],
        "layers": [2, 4, 3],
        "expansion_ratio": 4,
    },
}


def make_divisible(v: int, divisor: int = 8) -> int:
    """
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return int(new_v)


def get_norm_layer(num_features: int, norm_type: str):
    norm_type = norm_type.lower()

    if norm_type in ["batch_norm", "batch_norm_2d"]:
        norm_layer = nn.BatchNorm2d(
            num_features=num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True
        )
    elif norm_type in ["layer_norm", "ln"]:
        norm_layer = nn.LayerNorm(num_features, eps=1e-5, elementwise_affine=True)
    else:  # norm_type == 'identity':
        norm_layer = Identity()
    return norm_layer


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class GlobalPool(nn.Module):
    def __init__(self, keep_dim=False):
        super(GlobalPool, self).__init__()
        self.pool_type = "mean"
        self.keep_dim = keep_dim

    def forward(self, x):
        return torch.mean(x, dim=[-2, -1], keepdim=self.keep_dim)

    def __repr__(self):
        return "{}(type={})".format(self.__class__.__name__, self.pool_type)


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=False,
        norm_type="batch_norm_2d",
        use_norm=True,
        use_act=True,
    ):
        """
        Applies a 2D convolution over an input signal composed of several input planes.
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param kernel_size: kernel size
        :param stride: move the kernel by this amount during convolution operation
        :param dilation: Add zeros between kernel elements to increase the effective receptive field of the kernel.
        :param groups: Number of groups. If groups=in_channels=out_channels, then it is a depth-wise convolution
        :param bias: Add bias or not
        :param use_norm: Use normalization layer after convolution layer or not. Default is True.
        :param use_act: Use activation layer after convolution layer/convolution layer followed by batch
        normalization or not. Default is True.
        """
        super(ConvLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.bias = bias
        self.dilation = dilation

        if use_norm:  # True ==> bias == False
            assert not bias, "Do not use bias when using normalization layers."

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if isinstance(stride, int):
            stride = (stride, stride)

        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        assert isinstance(kernel_size, (tuple, list))
        assert isinstance(stride, (tuple, list))
        assert isinstance(dilation, (tuple, list))
        padding = (int((kernel_size[0] - 1) / 2) * dilation[0], int((kernel_size[1] - 1) / 2) * dilation[1])

        block = nn.Sequential()
        conv_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode="zeros",
        )
        block.add_module(name="conv", module=conv_layer)
        self.kernel_size = conv_layer.kernel_size

        self.norm_name = None
        if use_norm:  # True
            norm_layer = get_norm_layer(norm_type=norm_type, num_features=out_channels)
            block.add_module(name="norm", module=norm_layer)
            self.norm_name = norm_layer.__class__.__name__

        if use_act:  # True
            block.add_module(name="act", module=Swish())
            self.act_name = "Swish"

        self.block = block

    def forward(self, x):
        return self.block(x)

    def __repr__(self):
        repr_str = self.block[0].__repr__()
        repr_str = repr_str[:-1]

        if self.norm_name is not None:
            repr_str += ", normalization={}".format(self.norm_name)

        if self.act_name is not None:
            repr_str += ", activation={}".format(self.act_name)
        repr_str += ", bias={})".format(self.bias)
        return repr_str


class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        """
        Applies a linear transformation to the input data

        :param in_features: size of each input sample
        :param out_features:  size of each output sample
        :param bias: Add bias (learnable) or not
        """
        super(LinearLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        self.in_features = in_features
        self.out_features = out_features
        # self.reset_params()

    def forward(self, x):
        if self.bias is not None and x.dim() == 2:
            x = torch.addmm(self.bias, x, self.weight.t())
        else:
            x = x.matmul(self.weight.t())
            if self.bias is not None:
                x += self.bias
        return x

    def __repr__(self):
        repr_str = "{}(in_features={}, out_features={}, bias={})".format(
            self.__class__.__name__, self.in_features, self.out_features, True if self.bias is not None else False
        )
        return repr_str


class InvertedResidual(nn.Module):
    """
    Inverted residual block (MobileNetv2): https://arxiv.org/abs/1801.04381
    """

    def __init__(self, in_channels, out_channels, stride, expand_ratio, dilation=1):
        assert stride in [1, 2]
        super(InvertedResidual, self).__init__()
        self.stride = stride

        hidden_dim = make_divisible(int(round(in_channels * expand_ratio)), 8)
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        block = nn.Sequential()
        if expand_ratio != 1:
            block.add_module(
                name="exp_1x1",
                module=ConvLayer(
                    in_channels=in_channels, out_channels=hidden_dim, kernel_size=1, use_act=True, use_norm=True
                ),
            )

        block.add_module(
            name="conv_3x3",
            module=ConvLayer(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                stride=stride,
                kernel_size=3,
                groups=hidden_dim,
                use_act=True,
                use_norm=True,
                dilation=dilation,
            ),
        )

        block.add_module(
            name="red_1x1",
            module=ConvLayer(
                in_channels=hidden_dim, out_channels=out_channels, kernel_size=1, use_act=False, use_norm=True
            ),
        )

        self.block = block
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.exp = expand_ratio
        self.dilation = dilation

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)

    def __repr__(self) -> str:
        return "{}(in_channels={}, out_channels={}, stride={}, exp={}, dilation={})".format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.stride, self.exp, self.dilation
        )


class MultiHeadAttention(nn.Module):
    """
    This layer applies a multi-head attention as described in "Attention is all you need" paper
    https://arxiv.org/abs/1706.03762
    """

    def __init__(self, embed_dim, num_heads, bias=True):
        """
        :param embed_dim: Embedding dimension
        :param num_heads: Number of attention heads
        :param bias: Bias
        """
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Got: embed_dim={} and num_heads={}".format(embed_dim, num_heads)

        self.qkv_proj = LinearLayer(in_features=embed_dim, out_features=3 * embed_dim)

        self.attn_dropout = nn.Dropout(p=0.0, inplace=True)
        self.out_proj = LinearLayer(in_features=embed_dim, out_features=embed_dim)

        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.num_heads = num_heads
        self.embed_dim = embed_dim

    def forward(self, x):
        # [B x N x C]
        b_sz, n_patches, in_channels = x.shape

        # [B x N x C] --> [B x N x 3 x h x C]
        qkv = self.qkv_proj(x).reshape(b_sz, n_patches, 3, self.num_heads, -1)
        # [B x N x 3 x h x C] --> [B x h x 3 x N x C]
        qkv = qkv.transpose(1, 3)

        # [B x h x 3 x N x C] --> [B x h x N x C] x 3
        query, key, value = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        query = query * self.scaling

        # [B x h x N x C] --> [B x h x c x N]
        key = key.transpose(2, 3)

        # QK^T
        # [B x h x N x c] x [B x h x c x N] --> [B x h x N x N]
        attn = torch.matmul(query, key)
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)

        # weighted sum
        # [B x h x N x N] x [B x h x N x c] --> [B x h x N x c]
        out = torch.matmul(attn, value)

        # [B x h x N x c] --> [B x N x h x c] --> [B x N x C=ch]
        out = out.transpose(1, 2).reshape(b_sz, n_patches, -1)
        out = self.out_proj(out)

        return out


class TransformerEncoder(nn.Module):
    """
    This class defines the Transformer encoder (pre-norm) as described in "Attention is all you need" paper
        https://arxiv.org/abs/1706.03762
    """

    def __init__(self, embed_dim, ffn_latent_dim, num_heads=8, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        self.pre_norm_mha = nn.Sequential(
            get_norm_layer(norm_type="layer_norm", num_features=embed_dim),
            MultiHeadAttention(embed_dim, num_heads, bias=True),
            nn.Dropout(p=dropout, inplace=True),
        )

        self.pre_norm_ffn = nn.Sequential(
            get_norm_layer(norm_type="layer_norm", num_features=embed_dim),
            LinearLayer(in_features=embed_dim, out_features=ffn_latent_dim),
            Swish(),
            nn.Dropout(p=0.1, inplace=True),
            LinearLayer(in_features=ffn_latent_dim, out_features=embed_dim),
            nn.Dropout(p=dropout, inplace=True),
        )
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_latent_dim

    def forward(self, x):
        # Multi-head attention
        x = x + self.pre_norm_mha(x)
        # Feed forward network
        x = x + self.pre_norm_ffn(x)
        return x


class MobileViTBlock(nn.Module):
    """
    MobileViT block: https://arxiv.org/abs/2110.02178?context=cs.LG
    """

    def __init__(
        self,
        in_channels,
        transformer_dim,
        ffn_dim,
        n_transformer_blocks=2,
        head_dim=32,
        dropout=0.1,
        patch_h=8,
        patch_w=8,
        conv_ksize=3,
        dilation=1,
    ):
        super(MobileViTBlock, self).__init__()

        conv_3x3_in = ConvLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=conv_ksize,
            stride=1,
            use_norm=True,
            use_act=True,
            dilation=dilation,
        )
        conv_1x1_in = ConvLayer(
            in_channels=in_channels,
            out_channels=transformer_dim,
            kernel_size=1,
            stride=1,
            use_norm=False,
            use_act=False,
        )

        conv_1x1_out = ConvLayer(
            in_channels=transformer_dim, out_channels=in_channels, kernel_size=1, stride=1, use_norm=True, use_act=True
        )
        conv_3x3_out = ConvLayer(
            in_channels=2 * in_channels,
            out_channels=in_channels,
            kernel_size=conv_ksize,
            stride=1,
            use_norm=True,
            use_act=True,
        )
        self.local_rep = nn.Sequential()
        self.local_rep.add_module(name="conv_3x3", module=conv_3x3_in)
        self.local_rep.add_module(name="conv_1x1", module=conv_1x1_in)

        assert transformer_dim % head_dim == 0
        num_heads = transformer_dim // head_dim

        ffn_dims = [ffn_dim] * n_transformer_blocks

        global_rep = [
            TransformerEncoder(
                embed_dim=transformer_dim, ffn_latent_dim=ffn_dims[block_idx], num_heads=num_heads, dropout=dropout
            )
            for block_idx in range(n_transformer_blocks)
        ]
        global_rep.append(get_norm_layer(norm_type="layer_norm", num_features=transformer_dim))
        self.global_rep = nn.Sequential(*global_rep)

        self.conv_proj = conv_1x1_out

        self.fusion = conv_3x3_out

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_w * self.patch_h

        self.cnn_in_dim = in_channels
        self.cnn_out_dim = transformer_dim
        self.n_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.attn_dropout = 0.0
        self.ffn_dropout = 0.0
        self.dilation = dilation
        self.n_blocks = n_transformer_blocks
        self.conv_ksize = conv_ksize

    def unfolding(self, feature_map) -> Tuple[Tensor, Dict[str, int]]:
        patch_w, patch_h = self.patch_w, self.patch_h
        patch_area = int(patch_w * patch_h)
        batch_size, in_channels, orig_h, orig_w = feature_map.shape

        new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
        new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)
        interpolate = 0
        if new_w != orig_w or new_h != orig_h:
            # Note: Padding can be done, but then it needs to be handled in attention function.
            feature_map = F.interpolate(feature_map, size=(new_h, new_w), mode="bilinear", align_corners=False)
            interpolate = 1

        # number of patches along width and height
        num_patch_w = new_w // patch_w  # n_w
        num_patch_h = new_h // patch_h  # n_h
        num_patches = num_patch_h * num_patch_w  # N

        # [B, C, H, W] --> [B * C * n_h, p_h, n_w, p_w]
        reshaped_fm = feature_map.reshape(batch_size * in_channels * num_patch_h, patch_h, num_patch_w, patch_w)
        # [B * C * n_h, p_h, n_w, p_w] --> [B * C * n_h, n_w, p_h, p_w]
        transposed_fm = reshaped_fm.transpose(1, 2)
        # [B * C * n_h, n_w, p_h, p_w] --> [B, C, N, P] where P = p_h * p_w and N = n_h * n_w
        reshaped_fm = transposed_fm.reshape(batch_size, in_channels, num_patches, patch_area)
        # [B, C, N, P] --> [B, P, N, C]
        transposed_fm = reshaped_fm.transpose(1, 3)
        # [B, P, N, C] --> [BP, N, C]
        patches = transposed_fm.reshape(batch_size * patch_area, num_patches, -1)

        info_dict = {
            # "orig_size": (orig_h, orig_w),
            "orig_height": orig_h,
            "orig_width": orig_w,
            "batch_size": batch_size,
            "interpolate": interpolate,
            "total_patches": num_patches,
            "num_patches_w": num_patch_w,
            "num_patches_h": num_patch_h,
        }

        return patches, info_dict

    def folding(self, patches, info_dict: Dict[str, int]):
        n_dim = patches.dim()
        assert n_dim == 3, "Tensor should be of shape BPxNxC. Got: {}".format(patches.shape)
        # [BP, N, C] --> [B, P, N, C]
        patches = patches.contiguous().view(info_dict["batch_size"], self.patch_area, info_dict["total_patches"], -1)

        batch_size, pixels, num_patches, channels = patches.size()
        num_patch_h = info_dict["num_patches_h"]
        num_patch_w = info_dict["num_patches_w"]

        # [B, P, N, C] --> [B, C, N, P]
        patches = patches.transpose(1, 3)

        # [B, C, N, P] --> [B*C*n_h, n_w, p_h, p_w]
        feature_map = patches.reshape(batch_size * channels * num_patch_h, num_patch_w, self.patch_h, self.patch_w)
        # [B*C*n_h, n_w, p_h, p_w] --> [B*C*n_h, p_h, n_w, p_w]
        feature_map = feature_map.transpose(1, 2)
        # [B*C*n_h, p_h, n_w, p_w] --> [B, C, H, W]
        feature_map = feature_map.reshape(batch_size, channels, num_patch_h * self.patch_h, num_patch_w * self.patch_w)
        if info_dict["interpolate"] > 0:
            feature_map = F.interpolate(
                feature_map,
                size=(info_dict["orig_height"], info_dict["orig_width"]),
                mode="bilinear",
                align_corners=False,
            )
        return feature_map

    def forward(self, x):
        res = x

        fm = self.local_rep(x)

        # convert feature map to patches
        patches, info_dict = self.unfolding(fm)

        # learn global representations
        patches = self.global_rep(patches)

        # [B x Patch x Patches x C] --> [B x C x Patches x Patch]
        fm = self.folding(patches=patches, info_dict=info_dict)
        fm = self.conv_proj(fm)
        fm = self.fusion(torch.cat((res, fm), dim=1))

        return fm

    def __repr__(self):
        repr_str = "{}(".format(self.__class__.__name__)
        repr_str += "\n\tconv_in_dim={}, conv_out_dim={}, dilation={}, conv_ksize={}".format(
            self.cnn_in_dim, self.cnn_out_dim, self.dilation, self.conv_ksize
        )
        repr_str += "\n\tpatch_h={}, patch_w={}".format(self.patch_h, self.patch_w)
        repr_str += (
            "\n\ttransformer_in_dim={}, transformer_n_heads={}, transformer_ffn_dim={}, dropout={}, "
            "ffn_dropout={}, attn_dropout={}, blocks={}".format(
                self.cnn_out_dim,
                self.n_heads,
                self.ffn_dim,
                self.dropout,
                self.ffn_dropout,
                self.attn_dropout,
                self.n_blocks,
            )
        )

        repr_str += "\n)"
        return repr_str


class MobileViT(nn.Module):
    def __init__(self, features_list, dims_list, transformer_layers, expansion, image_size=256, num_classes=1000):
        super(MobileViT, self).__init__()

        self.image_size = image_size
        self.num_classes = num_classes
        self.dilation = 1

        self.conv_1 = ConvLayer(
            in_channels=3,
            out_channels=features_list[0],
            kernel_size=3,
            stride=2,
            use_norm=True,
            norm_type="batch_norm_2d",
            use_act=True,
        )
        # self.conv_1 -- Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2),
        #     padding=(1, 1), bias=False, normalization=BatchNorm2d, activation=Swish, bias=False)

        self.layer_1 = nn.Sequential(
            InvertedResidual(
                in_channels=features_list[0], out_channels=features_list[1], stride=1, expand_ratio=expansion
            )
        )
        # Sequential(
        #   (0): InvertedResidual(in_channels=16, out_channels=32, stride=1, exp=4, dilation=1)
        # )

        self.layer_2 = nn.Sequential(
            InvertedResidual(
                in_channels=features_list[1], out_channels=features_list[2], stride=2, expand_ratio=expansion
            ),
            InvertedResidual(
                in_channels=features_list[2], out_channels=features_list[2], stride=1, expand_ratio=expansion
            ),
            InvertedResidual(
                in_channels=features_list[2], out_channels=features_list[3], stride=1, expand_ratio=expansion
            ),
        )
        # Sequential(
        #   (0): InvertedResidual(in_channels=32, out_channels=64, stride=2, exp=4, dilation=1)
        #   (1): InvertedResidual(in_channels=64, out_channels=64, stride=1, exp=4, dilation=1)
        #   (2): InvertedResidual(in_channels=64, out_channels=64, stride=1, exp=4, dilation=1)
        # )

        self.layer_3 = nn.Sequential(
            InvertedResidual(
                in_channels=features_list[3], out_channels=features_list[4], stride=2, expand_ratio=expansion
            ),
            MobileViTBlock(
                in_channels=features_list[4],
                transformer_dim=dims_list[0],
                ffn_dim=dims_list[0] * 2,
                n_transformer_blocks=transformer_layers[0],
                head_dim=dims_list[0] // 4,
                patch_h=2,
                patch_w=2,
            ),
        )
        # Sequential(
        #   (0): InvertedResidual(in_channels=64, out_channels=96, stride=2, exp=4, dilation=1)
        #   (1): MobileViTBlock(
        #     conv_in_dim=96, conv_out_dim=144, dilation=1, conv_ksize=3
        #     patch_h=2, patch_w=2
        #     transformer_in_dim=144, transformer_n_heads=4, transformer_ffn_dim=288, dropout=0.1,
        #     blocks=2
        #   )
        # )

        self.layer_4 = nn.Sequential(
            InvertedResidual(
                in_channels=features_list[5], out_channels=features_list[6], stride=2, expand_ratio=expansion
            ),
            MobileViTBlock(
                in_channels=features_list[6],
                transformer_dim=dims_list[1],
                ffn_dim=dims_list[1] * 2,
                n_transformer_blocks=transformer_layers[1],
                head_dim=dims_list[1] // 4,
                patch_h=2,
                patch_w=2,
            ),
        )
        # Sequential(
        #   (0): InvertedResidual(in_channels=96, out_channels=128, stride=2, exp=4, dilation=1)
        #   (1): MobileViTBlock(
        #     conv_in_dim=128, conv_out_dim=192, dilation=1, conv_ksize=3
        #     patch_h=2, patch_w=2
        #     transformer_in_dim=192, transformer_n_heads=4, transformer_ffn_dim=384, dropout=0.1,
        #     blocks=4
        #   )
        # )

        self.layer_5 = nn.Sequential(
            InvertedResidual(
                in_channels=features_list[7], out_channels=features_list[8], stride=2, expand_ratio=expansion
            ),
            MobileViTBlock(
                in_channels=features_list[8],
                transformer_dim=dims_list[2],
                ffn_dim=dims_list[2] * 2,
                n_transformer_blocks=transformer_layers[2],
                head_dim=dims_list[2] // 4,
                patch_h=2,
                patch_w=2,
            ),
        )
        # Sequential(
        #   (0): InvertedResidual(in_channels=128, out_channels=160, stride=2, exp=4, dilation=1)
        #   (1): MobileViTBlock(
        #     conv_in_dim=160, conv_out_dim=240, dilation=1, conv_ksize=3
        #     patch_h=2, patch_w=2
        #     transformer_in_dim=240, transformer_n_heads=4, transformer_ffn_dim=480, dropout=0.1,
        #     blocks=3
        #   )
        # )

        self.conv_1x1_exp = ConvLayer(
            in_channels=features_list[9],
            out_channels=features_list[10],
            kernel_size=1,
            stride=1,
            use_act=True,
            norm_type="batch_norm_2d",
            use_norm=True,
        )
        # Conv2d(160, 640, kernel_size=(1, 1), stride=(1, 1), bias=False, normalization=BatchNorm2d,
        # activation=Swish, bias=False)

        self.classifier = nn.Sequential()
        self.classifier.add_module(name="global_pool", module=GlobalPool(keep_dim=False))
        self.classifier.add_module(name="dropout", module=nn.Dropout(p=0.1, inplace=True))
        self.classifier.add_module(
            name="fc", module=LinearLayer(in_features=features_list[10], out_features=self.num_classes)
        )
        # Sequential(
        #   (global_pool): GlobalPool(type=mean)
        #   (dropout): Dropout(p=0.1, inplace=True)
        #   (fc): LinearLayer(in_features=640, out_features=1000)
        # )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)

        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.conv_1x1_exp(x)
        x = self.classifier(x)
        return x


def MobileViT_XXS(pretrained=True):
    c = model_config["xxs"]
    model = MobileViT(c["features"], c["dims"], c["layers"], c["expansion_ratio"])
    if pretrained:
        model.load_state_dict(torch.load("models/mobilevit_xxs.pt", map_location="cpu"))

    return model


def MobileViT_XS(pretrained=True):
    c = model_config["xs"]
    model = MobileViT(c["features"], c["dims"], c["layers"], c["expansion_ratio"])
    if pretrained:
        model.load_state_dict(torch.load("models/mobilevit_xs.pt", map_location="cpu"))

    return model


def MobileViT_S(pretrained=True):
    c = model_config["s"]
    model = MobileViT(c["features"], c["dims"], c["layers"], c["expansion_ratio"])
    if pretrained:
        model.load_state_dict(torch.load("models/mobilevit_s.pt", map_location="cpu"))

    return model


def RefineMobileVit(model, new_class_numbers):
    input_features_numbers = model.classifier.fc.in_features
    model.classifier.fc = LinearLayer(in_features=input_features_numbers, out_features=new_class_numbers)


if __name__ == "__main__":
    img = torch.randn(1, 3, 256, 256)

    model_xxs = MobileViT_XXS()
    model_xs = MobileViT_XS()
    model_s = MobileViT_S()

    # XXS: 1.3M 、 XS: 2.3M 、 S: 5.6M
    print("XXS params: ", sum(p.numel() for p in model_xxs.parameters()))
    print(" XS params: ", sum(p.numel() for p in model_xs.parameters()))
    print("  S params: ", sum(p.numel() for p in model_s.parameters()))

    # print(model_s)

    model_s.cuda()
    model_s.eval()
    img = img.cuda()
    with torch.no_grad():
        output = model_s(img)

    print(output)

    model_s = torch.jit.script(model_s)
    pdb.set_trace()
