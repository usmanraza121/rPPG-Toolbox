
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import math
from functools import partial
# from torch import nn

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        try:
            self.act = Hardswish() if act else nn.Identity()
        except:
            self.act = nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

# ==============================
"""Replace depwise convolution"""
"""FPENet:Feature Pyramid Encoding Network for Real-time Semantic Segmentation"""
def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1, groups=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, dilation=dilation, groups=groups,bias=bias)

"""FPENet:Feature Pyramid Encoding Network for Real-time Semantic Segmentation"""
def conv1x1(in_planes, out_planes, stride=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)

"""FPENet:Feature Pyramid Encoding Network for Real-time Semantic Segmentation"""
class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, groups=2, padding=0 ) # group= 2 
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, groups=2, padding=0) # group= 2 
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.avg_pool(input)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return input * x
class FPEBlock(nn.Module):

    def __init__(self, inplanes, outplanes, dilat, downsample=None, stride=1, t=1, scales=4, se=True, norm_layer=None):
        super(FPEBlock, self).__init__()
        if inplanes % scales != 0:
            raise ValueError('Planes must be divisible by scales')
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        bottleneck_planes = inplanes * t
        self.conv1 = conv1x1(inplanes, bottleneck_planes, stride)
        self.bn1 = norm_layer(bottleneck_planes)
        self.conv2 = nn.ModuleList([conv3x3(bottleneck_planes // scales, bottleneck_planes // scales,
                                            groups=(bottleneck_planes // scales),dilation=dilat[i],
                                            padding=1*dilat[i]) for i in range(scales)])
        self.bn2 = nn.ModuleList([norm_layer(bottleneck_planes // scales) for _ in range(scales)])
        self.conv3 = conv1x1(bottleneck_planes, outplanes)
        self.bn3 = norm_layer(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEModule(outplanes) if se else None
        self.downsample = downsample
        self.stride = stride
        self.scales = scales

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        xs = torch.chunk(out, self.scales, 1)
        ys = []
        for s in range(self.scales):
            if s == 0:
                ys.append(self.relu(self.bn2[s](self.conv2[s](xs[s]))))
            else:
                ys.append(self.relu(self.bn2[s](self.conv2[s](xs[s] + ys[-1]))))
        out = torch.cat(ys, 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.se is not None:
            out = self.se(out)
            # print('SE module added')

        if self.downsample is not None:
            identity = self.downsample(identity)
            # print('identity module')

        out += identity
        out = self.relu(out)

        return out
# ===================================================
# FusedMBConv  https://avoid.overfit.cn/post/af49b27f50bb416ca829b4987e902874
class ResidualAdd(nn.Module):
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block
        
    def forward(self, x: Tensor) -> Tensor:
        res = x
        x = self.block(x)
        x += res
        return x


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int,
        norm: nn.Module = nn.BatchNorm2d,
        act: nn.Module = nn.ReLU,
        **kwargs
    ):

        super().__init__(
            nn.Conv2d(
                in_features,
                out_features,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            norm(out_features),
            act(),
        )

Conv1X1BnReLU = partial(ConvNormAct, kernel_size=1)
Conv3X3BnReLU = partial(ConvNormAct, kernel_size=3)
class FusedMBConv(nn.Sequential):
    def __init__(self, in_features: int, out_features: int, expansion: int = 4):
        residual = ResidualAdd if in_features == out_features else nn.Sequential
        expanded_features = in_features*expansion
        super().__init__(
            nn.Sequential(
                residual(
                    nn.Sequential(
                        Conv3X3BnReLU(in_features, 
                                      expanded_features, 
                                      act=nn.ReLU6
                                     ),
                        # here you can apply SE
                        # wide -> narrow
                        Conv1X1BnReLU(expanded_features, out_features, act=nn.Identity),
                    ),
                ),
                nn.ReLU(),
            )
        )

# =====================================


class Focus(nn.Module):
    # Focus wh information into c-space
    # slice concat conv
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))

# ======================================================
class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2



# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # CNN for spatial feature extraction (same as the previous one)
# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(ConvBlock, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.attention = AttentionBlock(in_channels)

#     def forward(self, x):
#         # Skip connection
#         skip = x
#         # Attention mechanism (before Conv2D)
#         attn = self.attention(x)
#         # Convolution and batch normalization
#         x = self.conv(attn)
#         x = self.bn(x)
#         x = F.relu(x)
#         # Adding skip connection to output
#         x = x + skip
#         return x

# class CNNSpatialExtractor(nn.Module):
#     def __init__(self, input_channels=3):
#         super(CNNSpatialExtractor, self).__init__()
#         self.layer1 = ConvBlock(input_channels, 64)  # 256x256 -> 128x128
#         self.layer2 = ConvBlock(64, 128)             # 128x128 -> 64x64
#         self.layer3 = ConvBlock(128, 256)            # 64x64 -> 32x32
#         self.layer4 = ConvBlock(256, 512)            # 32x32 -> 16x16
#         self.layer5 = ConvBlock(512, 1024)           # 16x16 -> 8x8

#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.layer5(x)
#         return x

# # LSTM to capture temporal dependencies
# class CNN_LSTM(nn.Module):
#     def __init__(self, input_channels=3, hidden_size=512, num_layers=1, num_classes=10):
#         super(CNN_LSTM, self).__init__()
#         self.cnn_extractor = CNNSpatialExtractor(input_channels)
        
#         # LSTM input size is based on the CNN output channels (1024) and the 8x8 feature map
#         self.lstm = nn.LSTM(1024 * 8 * 8, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, num_classes)

#     def forward(self, x):
#         batch_size, seq_length, C, H, W = x.size()
#         cnn_features = []
        
#         # Loop over the sequence of frames
#         for t in range(seq_length):
#             frame = x[:, t, :, :, :]  # Extract the t-th frame from the sequence
#             spatial_features = self.cnn_extractor(frame)
#             spatial_features = spatial_features.view(batch_size, -1)  # Flatten 8x8x1024 -> 65536
#             cnn_features.append(spatial_features)
        
#         # Stack CNN features for each frame: (batch_size, seq_length, feature_size)
#         cnn_features = torch.stack(cnn_features, dim=1)
        
#         # Pass through LSTM
#         lstm_out, _ = self.lstm(cnn_features)
        
#         # Take the last output of the LSTM (or use attention here for weighted output)
#         lstm_out = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
#         # Final classification layer
#         out = self.fc(lstm_out)
#         return out

# # Example usage:
# model = CNN_LSTM(input_channels=3, hidden_size=512, num_layers=2, num_classes=10)
# input_sequence = torch.randn(1, 10, 3, 256, 256)  # Batch size of 1, sequence of 10 frames, 3x256x256 each
# output = model(input_sequence)

# print(output.shape)  # Output shape: (batch_size, num_classes)


#=============================GSM Module https://github.com/swathikirans/GSM =====================
class gsmModule(nn.Module):
    def __init__(self, fPlane, num_segments=3):
        super(gsmModule, self).__init__()

        self.conv3D = nn.Conv3d(fPlane, 2, (3, 3, 3), stride=1,
                                padding=(1, 1, 1), groups=2)
        nn.init.constant_(self.conv3D.weight, 0)
        nn.init.constant_(self.conv3D.bias, 0)
        self.tanh = nn.Tanh()
        self.fPlane = fPlane
        self.num_segments = num_segments
        self.bn = nn.BatchNorm3d(num_features=fPlane)
        self.relu = nn.ReLU()

    def lshift_zeroPad(self, x):
        return torch.cat((x[:,:,1:], ftens(x.size(0), x.size(1), 1, x.size(3), x.size(4)).fill_(0)), dim=2)
    def rshift_zeroPad(self, x):
        return torch.cat((ftens(x.size(0), x.size(1), 1, x.size(3), x.size(4)).fill_(0), x[:,:,:-1]), dim=2)

    def forward(self, x):
        batchSize = x.size(0) // self.num_segments
        shape = x.size(1), x.size(2), x.size(3)
        assert  shape[0] == self.fPlane
        x = x.view(batchSize, self.num_segments, *shape).permute(0, 2, 1, 3, 4).contiguous()
        x_bn = self.bn(x)
        x_bn_relu = self.relu(x_bn)
        gate = self.tanh(self.conv3D(x_bn_relu))
        gate_group1 = gate[:, 0].unsqueeze(1)
        gate_group2 = gate[:, 1].unsqueeze(1)
        x_group1 = x[:, :self.fPlane // 2]
        x_group2 = x[:, self.fPlane // 2:]
        y_group1 = gate_group1 * x_group1
        y_group2 = gate_group2 * x_group2

        r_group1 = x_group1 - y_group1
        r_group2 = x_group2 - y_group2

        y_group1 = self.lshift_zeroPad(y_group1) + r_group1
        y_group2 = self.rshift_zeroPad(y_group2) + r_group2

        y_group1 = y_group1.view(batchSize, 2, self.fPlane // 4, self.num_segments, *shape[1:]).permute(0, 2, 1, 3, 4,
                                                                                                        5)
        y_group2 = y_group2.view(batchSize, 2, self.fPlane // 4, self.num_segments, *shape[1:]).permute(0, 2, 1, 3, 4,
                                                                                                        5)

        y = torch.cat((y_group1.contiguous().view(batchSize, self.fPlane//2, self.num_segments, *shape[1:]),
                       y_group2.contiguous().view(batchSize, self.fPlane//2, self.num_segments, *shape[1:])), dim=1)

        return y.permute(0, 2, 1, 3, 4).contiguous().view(batchSize*self.num_segments, *shape)


        #=================================

# Class WTSM(nn.Module):
#     def __init__(self, n_segment=10, fold_div=3, in_c=6,out_c=3):
#         super(WTSM, self).__init__()

#         self.in_channels = in_c
#         self.out_c3d = in_c*2
#         self.out_channels1d = out_c
#         self.n_segment = n_segment
#         self.fold_div = fold_div
#         self.conv3D = nn.Conv3d(in_c, self.out_c3d, (3, 3, 3), stride=1,
#                                 padding=(1, 1, 1))
#         # nn.init.constant_(self.conv3D.weight, 0)
#         # nn.init.constant_(self.conv3D.bias, 0)
        
#         nn.init.xavier_uniform_(self.conv3D.weight)
#         nn.init.constant_(self.conv3D.bias, 0)
#         # nn.init.kaiming_normal_(self.conv3D.weight, nonlinearity='relu')
#         # nn.init.constant_(self.conv3D.bias, 0)

#         self.conv1D = nn.Conv3d(in_c*2, out_c, (1, 1, 1), stride=1,
#                                 padding=0)

#         self.bn = nn.BatchNorm3d(num_features=self.out_c3d)
#         self.relu = nn.ReLU()
#         self.tanh = nn.Tanh()

#     def forward(self, x):
#         nt, c, h, w = x.size()
#         n_batch = nt // self.n_segment
#         x = x.view(n_batch, self.n_segment, c, h, w)
#         x1 = x.view(n_batch, self.n_segment, c, h, w).permute(0,2,1,3,4).contiguous()

#         fold = c // self.fold_div
#         out = torch.zeros_like(x)
#         out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
#         out[:, -1, :fold] = x[:, 0, :fold] # wrap left
#         out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
#         out[:, 0, fold: 2 * fold] = x[:, -1, fold: 2 * fold]  # wrap right
#         out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # no shift for final fold
#         x2= out.permute(0,2,1,3,4).contiguous()

#         y= torch.cat((x1,x2), dim=1)

#         x3 = self.conv3D(y)
#         x3 = self.bn(x3)
#         x3 = self.relu(x3)

#         x4 = self.conv1D(x3).permute(0,2,1,3,4).contiguous()
#         # print('x4', x4.shape)

#         y1 = x4.view(nt, self.out_channels1d, h, w)
#         # print('y1', y1.shape)
        
        
#         return y1
