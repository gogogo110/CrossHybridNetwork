from einops import rearrange
from copy import deepcopy
from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn
import torch
import numpy as np
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
import torch.nn.functional


import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_3tuple, trunc_normal_
from torch.autograd import Variable, Function


class DeformConv2D(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, bias=None):
        super(DeformConv2D, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv_kernel = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

    def forward(self, x, offset):
        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        # Change offset's order from [x1, x2, ..., y1, y2, ...] to [x1, y1, x2, y2, ...]
        # Codes below are written to make sure same results of MXNet implementation.
        # You can remove them, and it won't influence the module's performance.
        offsets_index = Variable(torch.cat([torch.arange(0, 2*N, 2), torch.arange(1, 2*N+1, 2)]), requires_grad=False).type_as(x).long()
        offsets_index = offsets_index.unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1).expand(*offset.size())
        offset = torch.gather(offset, dim=1, index=offsets_index)
        # ------------------------------------------------------------------------

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = Variable(p.data, requires_grad=False).floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], -1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], -1)

        # (b, h, w, N)
        mask = torch.cat([p[..., :N].lt(self.padding)+p[..., :N].gt(x.size(2)-1-self.padding),
                          p[..., N:].lt(self.padding)+p[..., N:].gt(x.size(3)-1-self.padding)], dim=-1).type_as(p)
        mask = mask.detach()
        floor_p = p - (p - torch.floor(p))
        p = p*(1-mask) + floor_p*mask
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv_kernel(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = np.meshgrid(range(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
                          range(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1), indexing='ij')
        # (2N, 1)
        p_n = np.concatenate((p_n_x.flatten(), p_n_y.flatten()))
        p_n = np.reshape(p_n, (1, 2*N, 1, 1))
        p_n = Variable(torch.from_numpy(p_n).type(dtype), requires_grad=False)

        return p_n

    @staticmethod
    def _get_p_0(h, w, N, dtype):
        p_0_x, p_0_y = np.meshgrid(range(1, h+1), range(1, w+1), indexing='ij')
        p_0_x = p_0_x.flatten().reshape(1, 1, h, w).repeat(N, axis=1)
        p_0_y = p_0_y.flatten().reshape(1, 1, h, w).repeat(N, axis=1)
        p_0 = np.concatenate((p_0_x, p_0_y), axis=1)
        p_0 = Variable(torch.from_numpy(p_0).type(dtype), requires_grad=False)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B,  H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., lmb=1):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        self.lmb=lmb
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_s = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_s, coords_h]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1

        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1

        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None, channel_weight=None):
        B_, N, C = x.shape  # B_是窗口的个数*bs
        
        qkv = self.qkv(x)  # B, N, 3C
        
        qkv=qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, self.num_heads, N, C // self.num_heads
        
        # 在这附近对q进行通道层面上的加权
        if channel_weight is not None:
            channel_weight = channel_weight.reshape(1, self.num_heads, 1, C // self.num_heads)
            q = q + q * channel_weight * self.lmb

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        # 在这附近得到每个窗口对应的注意力权重，想要得到64*7*7*1的形状
        # 在两个轴上求和，意味着对应每个像素输出的注意力权重和输入的注意力权重
        in_weight = attn.sum(dim=1).sum(-2).reshape(B_, self.window_size[0], self.window_size[1], 1)
        out_weight = attn.sum(dim=1).sum(-1).reshape(B_, self.window_size[0], self.window_size[1], 1)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return x, in_weight+out_weight

class MSABlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm,lmb=1):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.lmb = nn.Parameter(torch.tensor(lmb).float().cuda())
        # nn.init.ones_(m.bias) 
        # self.lmb = lmb
        self.shift_size = shift_size
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        
        self.attn = WindowAttention(
            dim, window_size=to_3tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, lmb=self.lmb)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.dwconv=nn.Conv2d(dim,dim,kernel_size=7,padding=3,groups=dim)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)


        self.SpatialConv = nn.Conv2d(1,1,kernel_size=7, padding=3)

        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.GMP = nn.AdaptiveMaxPool2d(1)
        self.MLP1 = nn.Linear(dim, dim//2)
        self.MLP2 = nn.Linear(dim//2, dim)



    def forward(self, x, mask_matrix):
        B, H,W, C = x.shape
        assert H * W==self.input_resolution[0]*self.input_resolution[1], "input feature has wrong size"
        
        shortcut = x
        x = self.norm1(x)

        # pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size

        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))  
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size,-self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
            
        # 考虑加入CBAM中对通道注意力中的处理思想, 获取通道注意力
        channel_gap = self.GAP(self.dwconv.weight) + self.dwconv.bias.reshape(-1, 1, 1, 1)
        channel_gap = self.MLP2(self.MLP1(channel_gap.reshape(1, 1, C))).reshape(C, 1, 1, 1)
        channel_gmp = self.GMP(self.dwconv.weight) + self.dwconv.bias .reshape(-1, 1, 1, 1)
        channel_gmp = self.MLP2(self.MLP1(channel_gmp.reshape(1, 1, C))).reshape(C, 1, 1, 1)
        channel_weight = torch.sigmoid(channel_gap + channel_gmp)
        # channel_weight = (channel_weight - torch.min(channel_weight)) / (torch.max(channel_weight) - torch.min(channel_weight))
        ########## SA分支 ##########
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows, attn_weight = self.attn(x_windows, mask=attn_mask, channel_weight=channel_weight)  # nW*B, window_size*window_size, C
        spatial_weight = window_reverse(attn_weight, self.window_size, Hp, Wp)
        spatial_weight = (spatial_weight - torch.min(spatial_weight)) / (torch.max(spatial_weight) - torch.min(spatial_weight))
        ########## SA分支 ##########
        spatial_weight = spatial_weight.permute(0, 3, 1, 2)
        spatial_weight = torch.sigmoid(self.SpatialConv(spatial_weight))
        spatial_weight = spatial_weight.permute(0, 2, 3, 1)

        ########## dwconv分支 ##########
        # 我想对kernel加权，等价于对feature加权
        spatial_shifted_x = shifted_x + shifted_x * spatial_weight * self.lmb
        dw = spatial_shifted_x.permute(0,3,1,2).contiguous()  # 转成正常的卷积输入
        dw = self.dwconv(dw)
        dw = dw.permute(0,2,3,1).contiguous()       # 转成正常的自注意力输入
        dw = window_partition(dw, self.window_size)  # nW*B, window_size, window_size, C
        dw = dw.view(-1, self.window_size * self.window_size, C)
        ########## dwconv分支 ##########

        # 两个分支结合
        if dw is not None:
            attn_windows = attn_windows + dw
        attn_windows = self.proj(attn_windows)
        attn_windows = self.proj_drop(attn_windows)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0 :
            x = x[:, :H, :W, :].contiguous()

        #x = x.view(B,  H * W, C)
        x = shortcut + self.drop_path(x)
        return x

class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv2d(dim,dim*2,kernel_size=3,stride=2,padding=1)
        self.norm = norm_layer(dim)

    def forward(self, x, H, W):
        x = x.permute(0,2,3,1).contiguous()
        x = F.gelu(x)
        x = self.norm(x)
        x=x.permute(0,3,1,2)
        x=self.reduction(x)
        return x
        
class Patch_Expanding(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.norm = norm_layer(dim)
        self.up=nn.ConvTranspose2d(dim,dim//2,2,2)
    def forward(self, x, H, W):
        x = x.permute(0,2,3,1).contiguous()
        x = self.norm(x)
        x = x.permute(0,3,1,2)
        x = self.up(x)
        return x
        
class BasicLayer(nn.Module):
    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size=7,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=True,
                 lmb=1
                 ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.dim=dim
        self.lmb=lmb
        # build blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                i_block=i,
                lmb=self.lmb
                )
            for i in range(depth)])
       
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
      
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
        mask_windows = mask_windows.view(-1,
                                          self.window_size * self.window_size)  
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        
        for blk in self.blocks:
            x = blk(x,attn_mask)
    
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x,  H, W, x_down, Wh, Ww
        else:
            return x,  H, W, x, H, W

class BasicLayer_up(nn.Module):

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size=7,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 upsample=True,
                 lmb=1
                ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.dim=dim
        self.lmb = lmb
        # build blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                i_block=i,
                lmb=self.lmb)
            for i in range(depth)])
        self.Upsample = upsample(dim=2*dim, norm_layer=norm_layer)

    def forward(self, x,skip, H, W):
        x_up = self.Upsample(x, H, W)
        x = x_up + skip
        H, W = H * 2, W * 2
        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
       
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
        mask_windows = mask_windows.view(-1,
                                          self.window_size * self.window_size)  
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:         
            x = blk(x,attn_mask)
            
        return x, H, W
        
class project(nn.Module):
    def __init__(self,in_dim,out_dim,stride,padding,activate,norm,last=False):
        super().__init__()
        self.out_dim=out_dim
        self.conv1=nn.Conv2d(in_dim,out_dim,kernel_size=3,stride=stride,padding=padding)
        self.conv2=nn.Conv2d(out_dim,out_dim,kernel_size=3,stride=1,padding=1)
        self.activate=activate()
        self.norm1=norm(out_dim)
        self.last=last  
        if not last:
            self.norm2=norm(out_dim)
            
    def forward(self,x):
        x=self.conv1(x)
        x=self.activate(x)
        #norm1
        Wh, Ww = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm1(x)
        x = x.transpose(1, 2).view(-1, self.out_dim, Wh, Ww)
        x=self.conv2(x)
        if not self.last:
            x=self.activate(x)
            #norm2
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm2(x)
            x = x.transpose(1, 2).view(-1, self.out_dim, Wh, Ww)
        return x

class project_up(nn.Module):
    def __init__(self,in_dim,out_dim,activate,norm,last=False):
        super().__init__()
        self.out_dim=out_dim
        self.conv1=nn.ConvTranspose2d(in_dim,out_dim,kernel_size=2,stride=2)
        self.conv2=nn.Conv2d(out_dim,out_dim,kernel_size=3,stride=1,padding=1)
        self.activate=activate()
        self.norm1=norm(out_dim)
        self.last=last  
        if not last:
            self.norm2=norm(out_dim)
            
    def forward(self,x):
        x=self.conv1(x)
        x=self.activate(x)
        #norm1
        Wh, Ww = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm1(x)
        x = x.transpose(1, 2).view(-1, self.out_dim, Wh, Ww)
        

        x=self.conv2(x)
        if not self.last:
            x=self.activate(x)
            #norm2
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm2(x)
            x = x.transpose(1, 2).view(-1, self.out_dim, Wh, Ww)
        return x
        
    

class PatchEmbed(nn.Module):

    def __init__(self, patch_size=4, in_chans=4, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_block=int(np.log2(patch_size[0]))
        self.project_block=[]
        self.dim=[int(embed_dim)//(2**i) for i in range(self.num_block)]
        self.dim.append(in_chans)
        self.dim=self.dim[::-1] # in_ch, embed_dim/2, embed_dim or in_ch, embed_dim/4, embed_dim/2, embed_dim
        
        for i in range(self.num_block)[:-1]:
            self.project_block.append(project(self.dim[i],self.dim[i+1],2,1,nn.GELU,nn.LayerNorm,False))
        self.project_block.append(project(self.dim[-2],self.dim[-1],2,1,nn.GELU,nn.LayerNorm,True))
        self.project_block=nn.ModuleList(self.project_block)

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, self.patch_size[0] - W % self.patch_size[0]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        for blk in self.project_block:
            x = blk(x)
       
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
            
class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, input_resolution=None,num_heads=None,window_size=None,i_block=None,qkv_bias=None,qk_scale=None,lmb=1):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.blocks_tr = MSABlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i_block % 2 == 0) else window_size // 2,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=0,
                attn_drop=0,
                drop_path=drop_path,
                lmb=lmb)
            
    def forward(self, x,mask):
        # 这里开始共享dwconv块和2-layer MLP
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        
        # 这里开始并行分支
        x = x.permute(0,2,3,1).contiguous()
        x = self.blocks_tr(x,mask)
        x = x.permute(0,3,1,2).contiguous()

        return x

class encoder(nn.Module):
    def __init__(self,
                 pretrain_img_size=[224,224],
                 patch_size=[4,4],
                 in_chans=1  ,
                 embed_dim=96,
                 depths=[3, 3, 3, 3],
                 num_heads=[3, 6, 12, 24],
                 window_size=[7,7,14,7],
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 lmb=1
                 ):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.lmb = lmb

        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(
                    pretrain_img_size[0] // patch_size[0] // 2 ** i_layer, pretrain_img_size[1] // patch_size[1] // 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size[i_layer],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_path=dpr[sum(
                    depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging
                if (i_layer < self.num_layers - 1) else None,
                lmb = self.lmb
                )
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

  
    def forward(self, x):
        """Forward function."""
        x = self.patch_embed(x)
        down=[]
       
        Wh, Ww = x.size(2), x.size(3)
        
        x = self.pos_drop(x)
        
      
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out=x_out.permute(0,2,3,1)
                x_out = norm_layer(x_out)
                out = x_out.view(-1, H, W, self.num_features[i]).permute(0,3, 1, 2).contiguous()            
                down.append(out)
        return down


class decoder(nn.Module):
    def __init__(self,
                 pretrain_img_size,
                 embed_dim,
                 patch_size=[4,4],
                 depths=[3,3,3],
                 num_heads=[24,12,6],
                 window_size=[14,7,7],
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 lmb=1
                 ):
        super().__init__()
        
        self.num_layers = len(depths)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.lmb = lmb

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers)[::-1]:
            
            layer = BasicLayer_up(
                dim=int(embed_dim * 2 ** (len(depths)-i_layer-1)),
                input_resolution=(
                    pretrain_img_size[0] // patch_size[0] // 2 ** (len(depths)-i_layer-1), pretrain_img_size[1] // patch_size[1] // 2 ** (len(depths)-i_layer-1)),
               
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size[i_layer],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_path=dpr[sum(
                    depths[:(len(depths)-i_layer-1)]):sum(depths[:(len(depths)-i_layer)])],
                norm_layer=norm_layer,
                upsample=Patch_Expanding,
                lmb = self.lmb
                )
            self.layers.append(layer)
        self.num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
    def forward(self,x,skips):
        outs=[]
        H, W = x.size(2), x.size(3)     
        x = self.pos_drop(x)

        for i in range(self.num_layers)[::-1]:          
            layer = self.layers[i]           
            x, H, W,  = layer(x,skips[i], H, W)
            outs.append(x)
        return outs

      
class final_patch_expanding(nn.Module):
    def __init__(self,dim,num_class,patch_size):
        super().__init__()
        self.num_block=int(np.log2(patch_size[0]))-2
        self.project_block=[]
        self.dim_list=[int(dim)//(2**i) for i in range(self.num_block+1)]
        # dim, dim/2, dim/4
        for i in range(self.num_block):
            self.project_block.append(project_up(self.dim_list[i],self.dim_list[i+1],nn.GELU,nn.LayerNorm,False))
        self.project_block=nn.ModuleList(self.project_block)
        self.up_final=nn.ConvTranspose2d(self.dim_list[-1],num_class,4,4)

    def forward(self,x):
        for blk in self.project_block:
            x = blk(x)
        x = self.up_final(x) 
        return x    
                                
class CxT_v1(SegmentationNetwork):
    def __init__(self, 
                 config, 
                 num_input_channels, 
                 embedding_dim, 
                 num_heads, 
                 num_classes, 
                 deep_supervision, 
                 conv_op=nn.Conv2d):
        super(CxT_v1, self).__init__()
        
        # Don't uncomment conv_op
        self.num_input_channels = num_input_channels
        self.num_classes = num_classes
        self.conv_op = conv_op
        self.do_ds = deep_supervision          
        self.embed_dim = embedding_dim
        self.depths=config.hyper_parameter.blocks_num
        self.num_heads=num_heads
        self.crop_size = config.hyper_parameter.crop_size
        self.patch_size=[config.hyper_parameter.convolution_stem_down,config.hyper_parameter.convolution_stem_down]
        self.window_size = config.hyper_parameter.window_size
        self.lmb = config.lmb
        # if window size of the encoder is [7,7,14,7], then decoder's is [14,7,7]. In short, reverse the list and start from the index of 1 
        self.model_down = encoder(
                                  pretrain_img_size=self.crop_size,
                                  window_size = self.window_size, 
                                  embed_dim=self.embed_dim,
                                  patch_size=self.patch_size,
                                  depths=self.depths,
                                  num_heads=self.num_heads,
                                  in_chans=self.num_input_channels,
                                  lmb = self.lmb
                                 )
                                        
        self.decoder = decoder(
                               pretrain_img_size=self.crop_size, 
                               window_size = self.window_size[::-1][1:],
                               embed_dim=self.embed_dim,
                               patch_size=self.patch_size,
                               depths=self.depths[::-1][1:],
                               num_heads=self.num_heads[::-1][1:],
                               lmb = self.lmb
                              )
   
        self.final=[]
        for i in range(len(self.depths)-1):
            self.final.append(final_patch_expanding(self.embed_dim*2**i,self.num_classes,patch_size=self.patch_size))
        self.final=nn.ModuleList(self.final)
        
    def forward(self, x):
        seg_outputs=[]
        skips = self.model_down(x)
        neck=skips[-1]
        out=self.decoder(neck,skips)
        
        for i in range(len(out)):  
            seg_outputs.append(self.final[-(i+1)](out[i]))
        if self.do_ds:
            # for training
            return seg_outputs[::-1]
            #size [[224,224],[112,112],[56,56]]

        else:
            #for validation and testing
            return seg_outputs[-1]
            #size [[224,224]]



        
        
if __name__ == '__main__':
    from thop import profile
    from nnunet.network_configuration.config import CONFIGS
    import time
    config = CONFIGS['ACDC_256']

    input = torch.randn(1, 4, 256, 256).cuda()
    network = CxT_v1(config, 
                        4, 
                        128, 
                        [4,8,16,32], 
                        2, 
                        None, 
                        conv_op=nn.Conv2d).cuda()
    T = 0
    for i in range(10):
        t1 = time.time()
        out = network(input)
        T += time.time()-t1
        print(time.time()-t1)
    print('avg', T/10)
    # flops, params = profile(network, (input,))
    # # print('out shape', out.shape)
    # print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
    
    
    # from thop import profile
    # flops, params = profile(network, (input,))
    # print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
    
    
    # from ptflops import get_model_complexity_info
    # flops, params = get_model_complexity_info(network, (4,256,256),as_strings=True,print_per_layer_stat=False)
    # print("flops: %s |params: %s" % (flops,params))


 