import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import to_2tuple, DropPath, trunc_normal_
from einops import rearrange
from .arch_restormer import CrossModal_TCAF_Block
import torchvision.transforms as transforms

def default_conv(in_channels, out_channels, kernel_size, bias=True, groups=1, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride=(stride, stride),
                     padding=(kernel_size // 2), bias=bias, groups=groups)

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 use_conv=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.use_conv = use_conv
        if use_conv:
            self.conv_layer = nn.Sequential(
                default_conv(in_channels=hidden_features, out_channels=hidden_features, kernel_size=3,
                             groups=hidden_features),
                nn.GELU()
            )

    def forward(self, x, x_size):
        # x: B, H * W, C
        # x_size: H, W
        H, W = x_size
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        if self.use_conv:
            x = rearrange(x, 'B (H W) C -> B C H W', H=H, W=W).contiguous()
            x = self.conv_layer(x)
            x = rearrange(x, 'B C H W -> B (H W) C').contiguous()
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 use_conv=False):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.use_conv = use_conv

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        if self.use_conv:
            self.qkv = default_conv(in_channels=dim, out_channels=dim * 3, kernel_size=3, bias=qkv_bias)
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        if self.use_conv:
            self.proj = default_conv(in_channels=dim, out_channels=dim, kernel_size=3)
        else:
            self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, x_size, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        H, W = x_size
        if self.use_conv:
            assert self.window_size[0] == self.window_size[1]
            x = window_reverse(x, self.window_size[0], H, W)  # (B, H, W, C)
            x = rearrange(x, 'B H W C -> B C H W').contiguous()
            x = self.qkv(x)
            x = rearrange(x, 'B C H W -> B H W C').contiguous()
            x = window_partition(x, self.window_size[0])  # num_windows*B, w, w, C
            qkv = rearrange(x, 'B w1 w2 C -> B (w1 w2) C').contiguous()  # num_windows*B, w*w, C
        else:
            qkv = self.qkv(x)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        if self.use_conv:
            x = window_reverse(x, self.window_size[0], H, W)  # (B, H, W, C)
            x = rearrange(x, 'B H W C -> B C H W').contiguous()
            x = self.proj(x)
            x = rearrange(x, 'B C H W -> B H W C').contiguous()
            x = window_partition(x, self.window_size[0])  # num_windows*B, w, w, C
            x = rearrange(x, 'B w1 w2 C -> B (w1 w2) C').contiguous()  # num_windows*B, w*w, C
        else:
            x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_mlp=True, use_conv=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.use_mlp = use_mlp
        self.mlp_ratio = mlp_ratio
        self.use_conv = use_conv
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, use_conv=use_conv)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.use_mlp:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                           use_conv=use_conv)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask, x_size=x_size)  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device), x_size=x_size)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H W C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        if self.use_mlp:
            x = x + self.drop_path(self.mlp(self.norm2(x), x_size))

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=2., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 use_mlp=True, use_conv=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 use_mlp=use_mlp,
                                 use_conv=use_conv)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        # x: B, H*W, C
        # x_size: H, W
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class RSTB(nn.Module):
    """
    Residual Swin Transformer Block (RSTB).
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 use_mlp=True, use_conv=False, ms=True):
        super(RSTB, self).__init__()
        self.window_size = window_size
        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint,
                                         use_mlp=use_mlp,
                                         use_conv=use_conv)
        self.ms = ms
        if ms:
            self.fuse_layer = default_conv(in_channels=3 * dim, out_channels=dim, kernel_size=1)
        self.last_layer = default_conv(in_channels=dim, out_channels=dim, kernel_size=3)

    def forward(self, x, x_size):
        # x: B, H*W, C
        # x_size: H, W
        if self.ms:
            H1, W1 = x_size
            # H2, W2 should be divisible by window_size
            H2, W2 = H1 // 2, W1 // 2
            H2, W2 = H2 - (H2 % self.window_size), W2 - (W2 % self.window_size)
            # H3, W3 should be divisible by window_size
            H3, W3 = H1 // 4, W1 // 4
            H3, W3 = H3 - (H3 % self.window_size), W3 - (W3 % self.window_size)

            x1 = rearrange(x, 'B (H W) C -> B C H W', H=H1, W=W1)
            x2 = F.interpolate(x1, size=(H2, W2), mode='bilinear', align_corners=False)
            x3 = F.interpolate(x1, size=(H3, W3), mode='bilinear', align_corners=False)

            x1 = rearrange(x1, 'B C H W -> B (H W) C')
            res1 = self.residual_group(x1, x_size=(H1, W1))  # B, H1*W1, C
            res1 = rearrange(res1, 'B (H W) C -> B C H W', H=H1, W=W1)

            x2 = rearrange(x2, 'B C H W -> B (H W) C')
            res2 = self.residual_group(x2, x_size=(H2, W2))  # B, H2*W2, C
            res2 = rearrange(res2, 'B (H W) C -> B C H W', H=H2, W=W2)
            res2 = F.interpolate(res2, size=(H1, W1), mode='bilinear', align_corners=False)

            x3 = rearrange(x3, 'B C H W -> B (H W) C')
            res3 = self.residual_group(x3, x_size=(H3, W3))  # B, H3*W3, C
            res3 = rearrange(res3, 'B (H W) C -> B C H W', H=H3, W=W3)
            res3 = F.interpolate(res3, size=(H1, W1), mode='bilinear', align_corners=False)

            res = torch.cat([res1, res2, res3], dim=1)
            res = self.last_layer(self.fuse_layer(res))
            res = rearrange(res, 'B C H W -> B (H W) C')

            return x + res
        else:
            H, W = x_size
            res = self.residual_group(x, x_size)  # B, H*W, C
            res = rearrange(res, 'B (H W) C -> B C H W', H=H, W=W)
            res = self.last_layer(res)
            res = rearrange(res, 'B C H W -> B (H W) C')

            return x + res
    
class TinySwinIR(nn.Module):
    def __init__(self, img_size=128, in_chs=65, out_chs=1, embed_dim=60, depths=(6,6,6,6), num_heads=(6,6,6,6),
                 window_size=8, mlp_ratio=2, norm_layer=nn.LayerNorm, use_mlp=True, use_conv=False, num_mid=1,
                 in_num=1, skip=True, anchor_num_layers=4, ms=False):
        # 加了独特参数，一般swinir中没有的
        # use_conv 和 use_mlp （这个应该没啥）
        # num_mid 是指重构两边的帧
        # anchor_num_layers指的是把6个块，从中间分开为3+3个块
        # ms 指多尺度
        super(TinySwinIR, self).__init__()
        self.window_size = window_size
        self.num_mid = num_mid
        self.in_num = in_num
        self.skip = skip
        self.anchor_num_layers = anchor_num_layers

        #  shallow feature extraction
        self.conv_down = nn.Sequential(default_conv(in_chs, embed_dim // in_num, kernel_size=3, stride=2))
        self.linear_reduce = nn.Linear(embed_dim + 1, embed_dim)

        # deep feature extraction
        self.num_layers = len(depths)  # number of Swin basic layers
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.layers = nn.ModuleList()
        Ph = Pw = img_size // window_size
        input_resolution = (Ph, Pw)
        for i in range(self.num_layers):
            self.layers.append(
                RSTB(
                    dim=embed_dim, input_resolution=input_resolution, depth=depths[i], num_heads=num_heads[i],
                    window_size=window_size, mlp_ratio=mlp_ratio, norm_layer=nn.LayerNorm, use_checkpoint=False,
                    use_mlp=use_mlp, use_conv=use_conv, ms=ms
                )
            )
        self.norm = norm_layer(self.embed_dim)

        # reconstruction
        self.mid_shuffle_up = nn.Sequential(
            default_conv(embed_dim, num_mid * out_chs * (4 ** 2), kernel_size=3, stride=1), nn.PixelShuffle(4))

        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        # initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def pad_image(self, x, pad_size):
        C, H, W = x.size()[-3:]
        pad_h = (pad_size - H % pad_size) % pad_size
        pad_w = (pad_size - W % pad_size) % pad_size
        if len(x.shape) == 5:
            N = x.shape[1]
            x = F.pad(x.reshape(-1, C, H, W), (0, pad_w, 0, pad_h), 'reflect')
            x = x.reshape(-1, N, *x.shape[-3:])
        elif len(x.shape) == 4:
            x = F.pad(x, (0, pad_w, 0, pad_h), 'reflect')
        else:
            raise ValueError
        return x

    def forward(self, x):
        # x: B, C', H', W'
        # t: 0~1
        Hi, Wi = x.shape[-2:]
        x = self.pad_image(x, 4 * self.window_size)
        x = self.conv_down(x)
        H, W = x.shape[-2:]
        x_size = (H, W)
        res = x
        x = rearrange(x, 'B C H W -> B (H W) C')

        if self.anchor_num_layers > 0:
            for i in range(self.anchor_num_layers):
                layer = self.layers[i]
                x = layer(x, x_size)
        x = rearrange(x, 'B (H W) C -> B C H W', H=H, W=W)
        if self.skip:
            x = res + self.conv_after_body(x)
        x = self.mid_shuffle_up(x)
        x = x[:, :, :Hi*2, :Wi*2]

        return x
    
class CAMMM(nn.Module):
    # Contetn Aware Motion Magnitude Module
    def __init__(self, hard_thresh=0.1, mid_dim=32):
        super(CAMMM, self).__init__()
        self.hard_thresh = hard_thresh
        self.conv1 = nn.Conv2d(2,mid_dim,3,1,1)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=mid_dim+1, out_channels=mid_dim+1, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.conv2 = nn.Conv2d(mid_dim+1,1,1,1,0)

    def forward(self, blur, tfi):
        N,C,H,W = blur.shape
        tfi = F.interpolate(tfi, size=(H, W), mode='bilinear', align_corners=False)
        hard_mask = torch.abs(blur-tfi)
        hard_mask = (hard_mask-torch.min(hard_mask))/(torch.max(hard_mask)-torch.min(hard_mask))
        hard_mask[hard_mask<=self.hard_thresh] = 1.
        hard_mask[hard_mask!=1] = 0.
        fused = torch.cat([blur,tfi],1)
        fused = self.conv1(fused)
        fused = torch.cat([fused,hard_mask.float()],1)
        fused = fused*self.sca(fused)
        fused = self.conv2(fused)
        soft_mask = torch.sigmoid(fused)
        return soft_mask

class SpkDeblurNet(nn.Module):
    def __init__(self, D_img_size=256, S_img_size=128,
                 D_in_chs=3, S_in_chs=65,
                 D_out_chs=3, S_out_chs=1,
                 D_embed_dim=180, S_embed_dim=60,
                 D_depths=(6,6,6,6,6,6), S_depths=(6,6,6,6),
                 D_num_heads=(6,6,6,6,6,6), S_num_heads=(6,6,6,6),
                 D_window_size=8, S_window_size=8,
                 D_anchor_num_layers=3, S_anchor_num_layers=3,
                 norm_layer=nn.LayerNorm, use_mlp=True, use_conv=False,
                 skip=True, ms=False):
        super(SpkDeblurNet, self).__init__()
        self.D_window_size = D_window_size
        self.skip = skip
        self.D_anchor_num_layers = D_anchor_num_layers

        #  shallow feature extraction
        self.D_conv_down = nn.Sequential(default_conv(D_in_chs, D_embed_dim, kernel_size=3, stride=2),
                                       nn.GELU(),
                                       default_conv(D_embed_dim, D_embed_dim, kernel_size=3, stride=2))
        self.conv_masked_blur = nn.Sequential(default_conv(1, S_embed_dim, kernel_size=3, stride=2),
                                       nn.GELU(),
                                       default_conv(S_embed_dim, S_embed_dim, kernel_size=3, stride=2))
        self.D_conv_reduce = nn.Sequential(default_conv(D_embed_dim + S_embed_dim, D_embed_dim, kernel_size=3, stride=1))

        # deep feature extraction
        self.D_num_layers = len(D_depths)  # number of Swin basic layers
        self.D_embed_dim = D_embed_dim
        self.D_layers = nn.ModuleList()
        for i in range(self.D_num_layers):
            self.D_layers.append(
                RSTB(
                    dim=D_embed_dim, input_resolution=(D_img_size // D_window_size,D_img_size // D_window_size), depth=D_depths[i], num_heads=D_num_heads[i],
                    window_size=D_window_size, mlp_ratio=2, norm_layer=nn.LayerNorm, use_checkpoint=False,
                    use_mlp=use_mlp, use_conv=use_conv, ms=ms
                )
            )

        # reconstruction
        self.D_mid_shuffle_up = nn.Sequential(
            default_conv(D_embed_dim, D_out_chs * (4 ** 2), kernel_size=3, stride=1), nn.PixelShuffle(4))
        self.D_last_shuffle_up = nn.Sequential(default_conv(D_embed_dim, D_out_chs * (4 ** 2), kernel_size=3, stride=1),
                                             nn.PixelShuffle(4))
        ############
        self.S_window_size = S_window_size
        self.S_anchor_num_layers = S_anchor_num_layers

        #  shallow feature extraction
        self.S_conv_down = nn.Sequential(default_conv(S_in_chs, S_embed_dim, kernel_size=3, stride=2))
        self.S_conv_reduce = nn.Sequential(default_conv(S_embed_dim + S_embed_dim, S_embed_dim, kernel_size=3, stride=1))

        # deep feature extraction
        self.S_num_layers = len(S_depths)  # number of Swin basic layers
        self.S_embed_dim = S_embed_dim
        self.S_layers = nn.ModuleList()
        for i in range(self.S_num_layers):
            self.S_layers.append(
                RSTB(
                    dim=S_embed_dim, input_resolution=(S_img_size // S_window_size, S_img_size // S_window_size), depth=S_depths[i], num_heads=S_num_heads[i],
                    window_size=S_window_size, mlp_ratio=2, norm_layer=nn.LayerNorm, use_checkpoint=False,
                    use_mlp=use_mlp, use_conv=use_conv, ms=ms
                )
            )

        # reconstruction
        self.S_shuffle_up = nn.Sequential(
            default_conv(S_embed_dim,  S_out_chs * (4 ** 2), kernel_size=3, stride=1), nn.PixelShuffle(4))

        self.S_conv_after_body = nn.Conv2d(S_embed_dim, S_embed_dim, 3, 1, 1)
        ##############
        self.cross_attention_1 = CrossModal_TCAF_Block(S_embed_dim, S_embed_dim, 4, bias=False, LayerNorm_type='WithBias')
        self.cross_attention_2 = CrossModal_TCAF_Block(D_embed_dim, S_embed_dim, 4, bias=False, LayerNorm_type='WithBias')
        self.cammm = CAMMM()

        # initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def pad_image(self, x, pad_size):
        C, H, W = x.size()[-3:]
        pad_h = (pad_size - H % pad_size) % pad_size
        pad_w = (pad_size - W % pad_size) % pad_size
        if len(x.shape) == 5:
            N = x.shape[1]
            x = F.pad(x.reshape(-1, C, H, W), (0, pad_w, 0, pad_h), 'reflect')
            x = x.reshape(-1, N, *x.shape[-3:])
        elif len(x.shape) == 4:
            x = F.pad(x, (0, pad_w, 0, pad_h), 'reflect')
        else:
            raise ValueError
        return x
    
    def _apply_cross_attention_1(self, x, y, h, w):
        x = rearrange(x, 'B (H W) C -> B C H W', H=h, W=w).contiguous()
        y = rearrange(y, 'B (H W) C -> B C H W', H=h, W=w).contiguous()
        x = self.cross_attention_1(x,y)
        x = rearrange(x, 'B C H W -> B (H W) C').contiguous()
        return x
    
    def _apply_cross_attention_2(self, x, y, h, w):
        x = rearrange(x, 'B (H W) C -> B C H W', H=h, W=w).contiguous()
        y = rearrange(y, 'B (H W) C -> B C H W', H=h, W=w).contiguous()
        x = self.cross_attention_2(x,y)
        x = rearrange(x, 'B C H W -> B (H W) C').contiguous()
        return x
    
    def forward(self, ori_blur, spike, t, tfi, blur_gray):
        Hb, Wb = ori_blur.shape[-2:]
        # blur = self.pad_image(ori_blur, 4 * self.D_window_size)
        # blur = self.D_conv_down(blur)
        blur = self.D_conv_down(ori_blur)
        blur = self.pad_image(blur, 4 * self.D_window_size)
        Hb1, Wb1 = blur.shape[-2:]
        blur_size = (Hb1, Wb1)
        blur = rearrange(blur, 'B C H W -> B (H W) C').contiguous()

        if self.D_anchor_num_layers > 0:
            res = blur
            for i in range(self.D_anchor_num_layers):
                layer = self.D_layers[i]
                blur = layer(blur, blur_size)
            if self.skip:
                mid_blur_fea = res + blur
        quick_blur_fea = rearrange(mid_blur_fea, 'B (H W) C -> B C H W', H=Hb1, W=Wb1).contiguous() #TODO t嵌入
        quick_blur_fea = self.D_mid_shuffle_up(quick_blur_fea)
        quick_sharp = quick_blur_fea[:, :, :Hb, :Wb]

        ########### SR ####
        Hs, Ws = spike.shape[-2:]
        # spike = self.pad_image(spike, 4 * self.S_window_size)
        # spike = self.S_conv_down(spike)
        spike = self.S_conv_down(spike)
        spike = self.pad_image(spike, 4 * self.S_window_size)
        Hs1, Ws1 = spike.shape[-2:]
        spike_size = (Hs1, Ws1)
        spike = rearrange(spike, 'B C H W -> B (H W) C').contiguous()
        if self.S_anchor_num_layers > 0:
            res = spike
            for i in range(self.S_anchor_num_layers):
                layer = self.S_layers[i]
                mid_spike_fea = layer(spike, spike_size)
            if self.skip:
                mid_spike_fea = mid_spike_fea + res

        quick_sharp_gray = transforms.Grayscale()(quick_sharp)
        soft_mask = self.cammm(quick_sharp_gray,tfi)
        masked_blur_fea = self.conv_masked_blur(soft_mask*quick_sharp_gray)
        masked_blur_fea = self.pad_image(masked_blur_fea, 4 * self.S_window_size)
        masked_blur_fea = rearrange(masked_blur_fea, 'B C H W -> B (H W) C').contiguous()
        fusion_fea = self._apply_cross_attention_1(mid_spike_fea, masked_blur_fea, Hs1, Ws1)

        if self.S_anchor_num_layers > 0:
            res = rearrange(fusion_fea, 'B (H W) C -> B C H W', H=Hs1, W=Ws1).contiguous()
            for i in range(self.S_anchor_num_layers, self.S_num_layers):
                layer = self.S_layers[i]
                fusion_fea = layer(fusion_fea, spike_size)
            spike = rearrange(fusion_fea, 'B (H W) C -> B C H W', H=Hs1, W=Ws1).contiguous()
            if self.skip:
                spike = res + self.S_conv_after_body(spike)
        spike_recon = self.S_shuffle_up(spike)
        spike_recon = spike_recon[:, :, :Hs*2, :Ws*2]

        ########### Fuse ####
        refined_fea = self._apply_cross_attention_2(mid_blur_fea, fusion_fea, Hb1, Wb1)
        if self.D_anchor_num_layers > 0:
            res = refined_fea
            for i in range(self.D_anchor_num_layers, self.D_num_layers):
                layer = self.D_layers[i]
                refined_fea = layer(refined_fea, blur_size)
            if self.skip:
                blur = res + refined_fea
        blur = rearrange(blur, 'B (H W) C -> B C H W', H=Hb1, W=Wb1).contiguous()
        sharp = self.D_last_shuffle_up(blur)
        sharp = sharp[:, :, :Hb, :Wb]

        # return sharp, spike_recon, quick_sharp, torch.ones_like(spike_recon)
        return sharp, spike_recon, quick_sharp, soft_mask
        # return sharp, spike_recon