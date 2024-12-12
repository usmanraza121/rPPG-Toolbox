"""Temporal Shift Convolutional Attention Network (TS-CAN).
Multi-Task Temporal Shift Attention Networks for On-Device Contactless Vitals Measurement
NeurIPS, 2020
Xin Liu, Josh Fromm, Shwetak Patel, Daniel McDuff
"""
from torch.cuda import FloatTensor as ftens
import torch
import torch.nn as nn
# from neural_methods.models_usman.common import Conv
# from LSTM_hybrid import Conv, FusedMBConv, Focus, FPEBlock
from .LSTM_hybrid import Conv, FusedMBConv, Focus, FPEBlock, Bottleneck

class Attention_mask(nn.Module):
    def __init__(self):
        super(Attention_mask, self).__init__()

    def forward(self, x):
        xsum = torch.sum(x, dim=2, keepdim=True)
        xsum = torch.sum(xsum, dim=3, keepdim=True)
        xshape = tuple(x.size())
        return x / xsum * xshape[2] * xshape[3] * 0.5

    def get_config(self):
        """May be generated manually. """
        config = super(Attention_mask, self).get_config()
        return config


class TSM(nn.Module):
    def __init__(self, n_segment=10, fold_div=3):
        super(TSM, self).__init__()
        self.n_segment = n_segment
        self.fold_div = fold_div

    def forward(self, x):
        nt, c, h, w = x.size()
        # print(f"nt: {nt}, c: {c}, h: {h}, w: {w}")
        # print('Frame_Depth:', self.n_segment)
        n_batch = nt // self.n_segment
        # print('n_batch:', n_batch)

        x = x.view(n_batch, self.n_segment, c, h, w)
        fold = c // self.fold_div
        # print('x:',x.shape)
        # print('fold', fold)
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift
        return out.view(nt, c, h, w)

############ Wrapping Time Shift Module #############
class WTSM1(nn.Module):
    def __init__(self, n_segment=10, fold_div=3):
        super(WTSM1, self).__init__()
        self.n_segment = n_segment  #n_segment = 10
        self.fold_div = fold_div

    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)
        x1 = x
        fold = c // self.fold_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
        out[:, -1, :fold] = x[:, 0, :fold] # wrap left
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
        out[:, 0, fold: 2 * fold] = x[:, -1, fold: 2 * fold]  # wrap right
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # no shift for final fold
        y = out.view(nt, c, h, w)
        return out.view(nt, c, h, w)

#++++++++++++++Wrap shift module with 3D Conv++++++++++++++++++++++++++++++
class WTSM(nn.Module):
    def __init__(self, n_segment=10, fold_div=3, in_c=6,out_c=6):
        super(WTSM, self).__init__()

        self.in_channels = in_c
        self.out_c3d = in_c
        self.out_channels1d = out_c
        self.n_segment = n_segment
        self.fold_div = fold_div
        self.conv3D = nn.Conv3d(in_c, self.out_c3d, (3, 3, 3), stride=1,
                                padding=(1, 1, 1))
        # nn.init.constant_(self.conv3D.weight, 0)
        # nn.init.constant_(self.conv3D.bias, 0)
        
        nn.init.xavier_uniform_(self.conv3D.weight)
        nn.init.constant_(self.conv3D.bias, 0)
        # nn.init.kaiming_normal_(self.conv3D.weight, nonlinearity='relu')
        # nn.init.constant_(self.conv3D.bias, 0)

        self.conv1D = nn.Conv3d(in_c*2, out_c, (1, 1, 1), stride=1,
                                padding=0)

        self.bn = nn.BatchNorm3d(num_features=self.out_c3d)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        # self.FPE = FPEBlock(in_c, out_c, [1, 2, 4, 8])

    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)
        x1 = x.view(n_batch, self.n_segment, c, h, w).permute(0,2,1,3,4).contiguous()
        # print('x1:', x1.shape)
        # x1 = self.conv3D(x1)
        # print('x11:', x1.shape)
        x1 = x1.permute(0,2,1,3,4).contiguous()
        # print('x111:', x1.shape)
        x1 = x1.view(nt, c, h, w)
        # print('x121:', x1.shape)

        fold = c // self.fold_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
        out[:, -1, :fold] = x[:, 0, :fold] # wrap left
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
        out[:, 0, fold: 2 * fold] = x[:, -1, fold: 2 * fold]  # wrap right
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # no shift for final fold
        x2 = out.view(nt, c, h, w)
        # print('x2:', x2.shape)
        # x2= out.permute(0,2,1,3,4).contiguous()

        y= x2
        # y = self.bn(y)
        # y = self.relu(y)

        # x3 = self.conv3D(y)
        # # x3 = self.bn(x3)
        # # x3 = self.relu(x3)
        # x3 = x3.permute(0,2,1,3,4).contiguous()

        # # x4 = self.conv1D(x3).permute(0,2,1,3,4).contiguous()
        # # print('x3', x3.shape)

        # y1 = x3.view(nt, self.in_channels, h, w)
        # # print('y1', y1.shape)
        
        
        return y

#--------------GSM-------------------
# class gsm1(nn.Module):
#     def __init__(self, fPlane=6, num_segments=10):
#         super(gsm1, self).__init__()

#         self.conv3D = nn.Conv3d(fPlane, 2, (3, 3, 3), stride=1,
#                                 padding=(1, 1, 1), groups=2)
#         nn.init.constant_(self.conv3D.weight, 0)
#         nn.init.constant_(self.conv3D.bias, 0)
#         self.tanh = nn.Tanh()
#         self.fPlane = fPlane
#         self.num_segments = num_segments
#         self.bn = nn.BatchNorm3d(num_features=fPlane)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         nt, c, h, w =x.size()
#         batchSize = x.size(0) // self.num_segments
#         shape = x.size(1), x.size(2), x.size(3)
#         # assert  shape[0] == self.fPlane
#         x = x.view(batchSize, self.num_segments, *shape).permute(0, 2, 1, 3, 4).contiguous()
        

#         x_bn = self.bn(x)
#         x_bn_relu = self.relu(x_bn)
#         gate = self.tanh(self.conv3D(x_bn_relu))
#         gate_group1 = gate[:, 0].unsqueeze(1)
#         gate_group2 = gate[:, 1].unsqueeze(1)
#         x_group1 = x[:, :self.fPlane // 2]
#         x_group2 = x[:, self.fPlane // 2:]
#         y_group1 = gate_group1 * x_group1
#         y_group2 = gate_group2 * x_group2

#         r_group1 = x_group1 - y_group1
#         r_group2 = x_group2 - y_group2

#         y_group1 = self.lshift_zeroPad(y_group1) + r_group1
#         y_group2 = self.rshift_zeroPad(y_group2) + r_group2

#         y_group1 = (self.lshift_zeroPad(y_group1) + r_group1).permute(0,2,1,3,4).contiguous()
#         y_group2 = (self.rshift_zeroPad(y_group2) + r_group2).permute(0,2,1,3,4).contiguous()

#         y1 = torch.cat((y_group1,y_group2),dim=1)
#         y1 = y1.contiguous().view(nt,c,h,w)
#         return y1

#     def lshift_zeroPad(self, x):
#         # return torch.cat((x[:,:,1:], ftens(x.size(0), x.size(1), 1, x.size(3), x.size(4)).fill_(0)), dim=2)
#         zero_padding = torch.zeros(x.size(0), x.size(1), 1, x.size(3), x.size(4)).to(x.device)
#         return torch.cat((x[:,:,1:], zero_padding), dim=2)    
#     def rshift_zeroPad(self, x):
#         # return torch.cat((ftens(x.size(0), x.size(1), 1, x.size(3), x.size(4)).fill_(0), x[:,:,:-1]), dim=2)
#         zero_padding = torch.zeros(x.size(0), x.size(1), 1, x.size(3), x.size(4)).to(x.device)  # Ensure same device as x
#         return torch.cat((zero_padding, x[:,:,:-1]), dim=2)
#==========================gsm2===========
# class gsm(nn.Module):
#     def __init__(self, fPlane=10, num_segments=10, fold_div=3):
#         super(gsm, self).__init__()
#         self.fPlane = fPlane
#         self.num_segments = num_segments
#         self.fold_div = fold_div

#         # 3D Convolution for gate generation
#         self.conv3D = nn.Conv3d(fPlane, 2, (3, 3, 3), stride=1, padding=(1, 1, 1), groups=1)
#         nn.init.constant_(self.conv3D.weight, 0)
#         nn.init.constant_(self.conv3D.bias, 0)

#         self.tanh = nn.Tanh()
#         self.bn = nn.BatchNorm3d(num_features=fPlane)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         # Reshape input tensor to (batch, num_segments, channels, height, width)
#         nt, c, h, w = x.size()
#         n_batch = nt // self.num_segments
#         x = x.view(n_batch, self.num_segments, c, h, w)
        
#         # Fold-based shifting logic from WTSM
#         fold = c // self.fold_div
#         shifted = torch.zeros_like(x, device=x.device)
#         shifted[:, :-1, :fold] = x[:, 1:, :fold]  # Shift left
#         shifted[:, -1, :fold] = x[:, 0, :fold]  # Wrap left
#         shifted[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # Shift right
#         shifted[:, 0, fold: 2 * fold] = x[:, -1, fold: 2 * fold]  # Wrap right
#         shifted[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # No shift for final fold

#         # Rearrange for 3D BatchNorm and Conv3D
#         shifted = shifted.permute(0, 2, 1, 3, 4).contiguous()  # (batch, channels, segments, height, width)
#         shifted = self.bn(shifted)
#         shifted = self.relu(shifted)

#         # Generate gates with Conv3D and apply gating
#         gate = self.tanh(self.conv3D(shifted))
#         gate_group1, gate_group2 = gate[:, 0:1], gate[:, 1:2]  # Split gates into two groups
#         x_group1, x_group2 = shifted[:, :c // 2], shifted[:, c // 2:]  # Split input into two groups

#         y_group1 = gate_group1 * x_group1
#         y_group2 = gate_group2 * x_group2

#         # Residual connections for gated outputs
#         r_group1 = x_group1 - y_group1
#         r_group2 = x_group2 - y_group2

#         # Combine groups and apply fold-based shifts
#         y_group1 = self._lshift_zeroPad(y_group1) + r_group1
#         y_group2 = self._rshift_zeroPad(y_group2) + r_group2

#         # Concatenate and reshape to original format
#         y = torch.cat((y_group1, y_group2), dim=1)  # Combine groups
#         y = y.permute(0, 2, 1, 3, 4).contiguous()  # (batch, segments, channels, height, width)
#         return y.view(nt, c, h, w)

#     def _lshift_zeroPad(self, x):
#         """Left shift with zero padding."""
#         return torch.cat((x[:, :, 1:], torch.zeros_like(x[:, :, :1])), dim=2)

#     def _rshift_zeroPad(self, x):
#         """Right shift with zero padding."""
#         return torch.cat((torch.zeros_like(x[:, :, :1]), x[:, :, :-1]), dim=2)


#============================================================

class TSCAN(nn.Module):

    def __init__(self, in_channels=3, nb_filters1=32, nb_filters2=64, kernel_size=3, dropout_rate1=0.25,
                 dropout_rate2=0.5, pool_size=(2, 2), nb_dense=128, frame_depth=20, img_size=72):
        """Definition of TS_CAN.
        Args:
          in_channels: the number of input channel. Default: 3
          frame_depth: the number of frame (window size) used in temport shift. Default: 20
          img_size: height/width of each frame. Default: 36.
        Returns:
          TS_CAN model."""
        super(TSCAN, self).__init__()
        self.focus1e = Focus(6,32,3)  #0
        self.con1_1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=1, stride=1, padding=0)
        self.con1_2 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=1, stride=1, padding=0)
        self.bot3 = FusedMBConv(6,6) # 2

        self.conv1e = Conv(32,64,3,2)  # 1
        # self.bottleneck1e = Bottleneck(64,64) # 2
        self.bottleneck1e = FusedMBConv(64,64) # 2

        self.conv2e = Conv(64,128,3,2)  # 3
        # self.bottleneck2e = Bottleneck(128,128)  # 4
        self.bottleneck2e = FusedMBConv(128,128)  # 4

        self.conv3e = Conv(128,256,3,2)  #5
        # self.bottleneck3e = Bottleneck(256,256)   #6
        self.bottleneck3e = FusedMBConv(256,256)   #6

        self.conv4e = Conv(256,512,3,2) #7
        # self.sppe = SPP(512,512,[5,9,13]) #8

        # self.fpe = FPEBlock(512, 512, [1, 2, 4, 8])
        self.bottleneck4e = Bottleneck(512,512) # 9
        # self.bottleneck4e = FusedMBConv(512,512) # 9

        # ======================================================================
        # super(TSCAN, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.pool_size = pool_size
        self.nb_filters1 = nb_filters1
        self.nb_filters2 = nb_filters2
        self.nb_dense = nb_dense
        self.f_depth = frame_depth
        # TSM layers
        # self.TSM_1 = WTSM(in_c=3,out_c=3)
        # self.TSM_2 = WTSM(in_c=32,out_c=32)
        # self.TSM_3 = WTSM(in_c=32,out_c=32)
        # self.TSM_4 = WTSM(in_c=64,out_c=64)
        # self.TSM_5 = WTSM()
        # self.fpe2 = FPEBlock(32, 32, [1, 2, 4, 8])
        # self.fpe3 = FPEBlock(64, 64, [1, 2, 4, 8])
        self.fpe4 = FPEBlock(64, 64, [1, 4, 8, 32])      # in use @fpe2
        self.TSM_1 = WTSM1()  # my modified
        self.TSM_2 = WTSM1()
        self.TSM_3 = WTSM1()
        self.TSM_4 = WTSM1()

        # self.TSM_1 = TSM()  # original
        # self.TSM_2 = TSM()
        # self.TSM_3 = TSM()
        # self.TSM_4 = TSM()
        # self.TSM_6 = gsm1()
        # self.TSM_7 = gsm1()
        # self.TSM_8 = gsm1()
        # Motion branch convs
        self.motion_conv1 = nn.Conv2d(in_channels, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1),
                                      bias=True)
        self.motion_conv2 = nn.Conv2d(
            self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, bias=True)
        self.motion_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1),
                                      bias=True)
        self.motion_conv4 = nn.Conv2d(
            self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, bias=True)
        # Apperance branch convs
        self.apperance_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size,
                                         padding=(1, 1), bias=True)
        self.apperance_conv2 = nn.Conv2d(
            self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, bias=True)
        self.apperance_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size,
                                         padding=(1, 1), bias=True)
        self.apperance_conv4 = nn.Conv2d(
            self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, bias=True)
        # Attention layers
        self.apperance_att_conv1 = nn.Conv2d(
            self.nb_filters1, 1, kernel_size=1, padding=(0, 0), bias=True)
        self.attn_mask_1 = Attention_mask()
        self.apperance_att_conv2 = nn.Conv2d(
            self.nb_filters2, 1, kernel_size=1, padding=(0, 0), bias=True)
        self.attn_mask_2 = Attention_mask()
        # Avg pooling
        self.avg_pooling_1 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_2 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_3 = nn.AvgPool2d(self.pool_size)
        # Dropout layers
        self.dropout_1 = nn.Dropout(self.dropout_rate1)
        self.dropout_2 = nn.Dropout(self.dropout_rate1)
        self.dropout_3 = nn.Dropout(self.dropout_rate1)
        self.dropout_4 = nn.Dropout(self.dropout_rate2)
        # Dense layers
        if img_size == 36:
            self.final_dense_1 = nn.Linear(3136, self.nb_dense, bias=True)
        elif img_size == 72:
            self.final_dense_1 = nn.Linear(16384, self.nb_dense, bias=True)
        elif img_size == 96:
            self.final_dense_1 = nn.Linear(30976, self.nb_dense, bias=True)
        elif img_size == 128:
            self.final_dense_1 = nn.Linear(57600, self.nb_dense, bias=True)
        elif img_size == 144:
            self.final_dense_1 = nn.Linear(12800, self.nb_dense, bias=True)
        else:
            raise Exception('Unsupported image size')
        self.final_dense_2 = nn.Linear(self.nb_dense, 1, bias=True)
        self.final_dense = nn.Linear(512, self.nb_dense, bias=True)

        self.final_dense_3 = nn.Linear(3136, self.nb_dense, bias=True)

        # self.final_dense = nn.Linear(57600, self.nb_dense, bias=True)

    def forward(self, inputs, params=None):
        # print('inputss',inputs.size)
        # raw_input = inputs[:, :, :, :]             #    720, 6, 72, 72
        # # print('raw_input::', raw_input.shape)
        # # print('\n you are running me \n')

        # focuss1e = self.focus1e(raw_input)            #      720, 32, 36, 36
        # att_1 = self.TSM_1(focuss1e)

        # g1 = att_1 * focuss1e

        # conv1e = self.conv1e(g1)             # 1    720, 64, 18, 18
        # att_2 = self.TSM_2(conv1e)

        # bottleneck1e =self.bottleneck1e(att_2)    # 2    720, 64, 18, 18
        

        # # print('focuss1e:', focuss1e.shape)
        # # print('conv1e:', conv1e.shape)
        # # print('bottleneck1e:', bottleneck1e.shape)

        # conv2e =self.conv2e(bottleneck1e)          # 3     720, 128, 9, 9
        # att_3 = self.TSM_3(conv2e)
        # # print('conv2e:', conv2e.shape)

        # cnn_encoder_1 =self.bottleneck2e(att_3)   # 4     720, 128, 9, 9
        # # att_4 = self.TSM_4(diff_input)
        # # print('cnn_encoder-1:', cnn_encoder_1.shape)

        # conv3e =self.conv3e(cnn_encoder_1)         # 5     720, 256, 5, 5
        # att_5 = self.TSM_5(conv3e)
        # # print('conv3e:', conv3e.shape)

        # cnn_encoder_2 =self.bottleneck3e(att_5)   # 6     720, 256, 5, 5
        # # print('cnn-encoder2:', cnn_encoder_2.shape)

        # cnn_encoder_3 =self.conv4e(cnn_encoder_2)   # 7     720, 512, 3, 3
        # # print('cnn-encoder-3:', cnn_encoder_3.shape)
        # att_6 = self.TSM_6(cnn_encoder_3)

        # feature_n = self.fpe(cnn_encoder_3)         # 8     720, 512, 3, 3
        # # print('feature_n;', feature_n.shape)
        # # bottleneck4e =self.bottleneck4e(feature_n) 
        # # print('bottleneck4e:'. bottleneck4e.shape)
        
        # d7 = self.avg_pooling_3(feature_n)
        # # print('d7', d7.shape)
        # d8 = self.dropout_3(d7)
        # # print('d8', d8.shape)
        # d9 = d8.view(d8.size(0), -1)
        # # print('d9_shape:', d9.shape)
        # # d10 = torch.tanh(self.final_dense_1(d9))
        # d10 = torch.tanh(self.final_dense(d9))
        # # print('d10', d10.shape)
        # d11 = self.dropout_4(d10)
        # # print('d11', d11.shape)
        # out = self.final_dense_2(d11)
        # # print('out:', out.shape)

        # return out
# # =============================================original model========
        diff_input = inputs[:, :3, :, :]
        raw_input = inputs[:, 3:, :, :]

        # print('diff_input is:', diff_input.shape)
        # print('raw_input is:', raw_input.shape)
        # r1 = self.con1_2(diff_input)
        # diff_input = r1
        # raw_input = self.con1_1(raw_input)
        # r2 = bot3(r1)
        diff_input = self.TSM_1(diff_input)         #TSM1
        # print('TSM1', diff_input.shape)                   # [720, 3, 72, 72]
        d1 = torch.tanh(self.motion_conv1(diff_input))
        # print('before-d1', d1.shape)
        d1 = self.TSM_2(d1)                         #TSM2
        # print('after-d1_TSM2', d1.shape)                  # [720, 32, 72, 72]
        
        d2 = torch.tanh(self.motion_conv2(d1))      #Motion2
        # print('d2', d2.shape)                             # [720, 32, 72, 72]
        r1 = torch.tanh(self.apperance_conv1(raw_input)) #AP1 
        # print('r1', r1.shape)                             # [720, 32, 70, 70]
        r2 = torch.tanh(self.apperance_conv2(r1))   #AP2
        # r2 = self.fpe2(r2)                          #FPE1@AP
        # print('r2', r2.shape)                             # [720, 32, 70, 70]

        g1 = torch.sigmoid(self.apperance_att_conv1(r2))
        # print('g1', g1.shape)                             # [720, 1, 70, 70]
        g1 = self.attn_mask_1(g1)                   #Attention Mask 1
        # print('g1_1', g1.shape)                           # [720, 1, 70, 70]
        gated1 = d2 * g1
        # print('gated1', gated1.shape)                     # [720, 32, 70, 70]

        d3 = self.avg_pooling_1(gated1)
        # print('d3', d3.shape)                             # [720, 32, 35, 35]
        d4 = self.dropout_1(d3)
        # print('d4', d4.shape)                             # [720, 32, 35, 35]

        r3 = self.avg_pooling_2(r2)
        # print('r3', r3.shape)                             # [720, 32, 35, 35]
        r4 = self.dropout_2(r3)
        # print('r4', r4.shape)                             # [720, 32, 35, 35]

        d4 = self.TSM_3(d4)                         # TSM3
        # print('d4', d4.shape)                             # [720, 32, 35, 35]
        d5 = torch.tanh(self.motion_conv3(d4))      #Motion3
        # print('d5', d5.shape)                             # [720, 64, 35, 35]
        d5 = self.TSM_4(d5)                         #TSM4
        # print('d5', d5.shape)                             # [720, 64, 35, 35]
        d6 = torch.tanh(self.motion_conv4(d5))      #Motion4
        # print('d6', d6.shape)                             # [720, 64, 33, 33]

        r5 = torch.tanh(self.apperance_conv3(r4))   #AP3
        # print('r5', r5.shape)                             # [720, 64, 35, 35]
        r6 = torch.tanh(self.apperance_conv4(r5))   #AP4
        # print('r6', r6.shape)                             # [720, 64, 33, 33]
        r6 = self.fpe4(r6)                          #FPE2@AP
        g2 = torch.sigmoid(self.apperance_att_conv2(r6))
        # print('g2', g2.shape)                             # [720, 1, 33, 33]
        g2 = self.attn_mask_2(g2)
        # print('g2', g2.shape)                             # [720, 1, 33, 33]
        gated2 = d6 * g2
        # print('gated2', gated2.shape)                     # [720, 64, 33, 33]

        d7 = self.avg_pooling_3(gated2)
        # print('d7', d7.shape)                             # [720, 64, 16, 16]
        d8 = self.dropout_3(d7)
        
        # print('d8', d8.shape)                             # [720, 64, 16, 16]
        d9 = d8.view(d8.size(0), -1)
        # print('d9', d9.shape)                             # [720, 16384]
        d10 = torch.tanh(self.final_dense_1(d9))
        # print('d10', d10.shape)                           # [720, 128]
        d11 = self.dropout_4(d10)
        # print('d11', d11.shape)                           # [720, 128]
        out = self.final_dense_2(d11)
        # print('out-final', out.shape)                     # [720, 1]

        return out

'''
class MTTS_CAN(nn.Module):
    """MTTS_CAN is the multi-task (respiration) version of TS-CAN"""

    def __init__(self, in_channels=3, nb_filters1=32, nb_filters2=64, kernel_size=3, dropout_rate1=0.25,
                 dropout_rate2=0.5, pool_size=(2, 2), nb_dense=128, frame_depth=20):
        super(MTTS_CAN, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.pool_size = pool_size
        self.nb_filters1 = nb_filters1
        self.nb_filters2 = nb_filters2
        self.nb_dense = nb_dense
        # TSM layers
        self.TSM_1 = TSM(n_segment=frame_depth)
        self.TSM_2 = TSM(n_segment=frame_depth)
        self.TSM_3 = TSM(n_segment=frame_depth)
        self.TSM_4 = TSM(n_segment=frame_depth)
        # Motion branch convs
        self.motion_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1),
                                      bias=True)
        self.motion_conv2 = nn.Conv2d(
            self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, bias=True)
        self.motion_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1),
                                      bias=True)
        self.motion_conv4 = nn.Conv2d(
            self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, bias=True)
        # Apperance branch convs
        self.apperance_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size,
                                         padding=(1, 1), bias=True)
        self.apperance_conv2 = nn.Conv2d(
            self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, bias=True)
        self.apperance_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size,
                                         padding=(1, 1), bias=True)
        self.apperance_conv4 = nn.Conv2d(
            self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, bias=True)
        # Attention layers
        self.apperance_att_conv1 = nn.Conv2d(
            self.nb_filters1, 1, kernel_size=1, padding=(0, 0), bias=True)
        self.attn_mask_1 = Attention_mask()
        self.apperance_att_conv2 = nn.Conv2d(
            self.nb_filters2, 1, kernel_size=1, padding=(0, 0), bias=True)
        self.attn_mask_2 = Attention_mask()
        # Avg pooling
        self.avg_pooling_1 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_2 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_3 = nn.AvgPool2d(self.pool_size)
        # Dropout layers
        self.dropout_1 = nn.Dropout(self.dropout_rate1)
        self.dropout_2 = nn.Dropout(self.dropout_rate1)
        self.dropout_3 = nn.Dropout(self.dropout_rate1)
        self.dropout_4_y = nn.Dropout(self.dropout_rate2)
        self.dropout_4_r = nn.Dropout(self.dropout_rate2)

        # Dense layers
        self.final_dense_1_y = nn.Linear(16384, self.nb_dense, bias=True)
        self.final_dense_2_y = nn.Linear(self.nb_dense, 1, bias=True)
        self.final_dense_1_r = nn.Linear(16384, self.nb_dense, bias=True)
        self.final_dense_2_r = nn.Linear(self.nb_dense, 1, bias=True)

    def forward(self, inputs, params=None):
        diff_input = inputs[:, :3, :, :]
        raw_input = inputs[:, 3:, :, :]

        diff_input = self.TSM_1(diff_input)
        d1 = torch.tanh(self.motion_conv1(diff_input))
        d1 = self.TSM_2(d1)
        d2 = torch.tanh(self.motion_conv2(d1))

        r1 = torch.tanh(self.apperance_conv1(raw_input))
        r2 = torch.tanh(self.apperance_conv2(r1))

        g1 = torch.sigmoid(self.apperance_att_conv1(r2))
        g1 = self.attn_mask_1(g1)
        gated1 = d2 * g1

        d3 = self.avg_pooling_1(gated1)
        d4 = self.dropout_1(d3)

        r3 = self.avg_pooling_2(r2)
        r4 = self.dropout_2(r3)

        d4 = self.TSM_3(d4)
        d5 = torch.tanh(self.motion_conv3(d4))
        d5 = self.TSM_4(d5)
        d6 = torch.tanh(self.motion_conv4(d5))

        r5 = torch.tanh(self.apperance_conv3(r4))
        r6 = torch.tanh(self.apperance_conv4(r5))

        g2 = torch.sigmoid(self.apperance_att_conv2(r6))
        g2 = self.attn_mask_2(g2)
        gated2 = d6 * g2

        d7 = self.avg_pooling_3(gated2)
        d8 = self.dropout_3(d7)
        d9 = d8.view(d8.size(0), -1)

        d10 = torch.tanh(self.final_dense_1_y(d9))
        d11 = self.dropout_4_y(d10)
        out_y = self.final_dense_2_y(d11)

        d10 = torch.tanh(self.final_dense_1_r(d9))
        d11 = self.dropout_4_r(d10)
        out_r = self.final_dense_2_r(d11)

        return out_y, out_r
        '''


if __name__ == "__main__":
    from torchsummary import summary
    net = TSCAN()
    # print(net)
    print('\n:')
    t1 = torch.rand(720, 6, 72, 72)
    x2 = net(t1)
    print("input: ", t1.shape)
    # print("input: " + str(x2.shape))
    print('out:',x2.shape)
    # print(summary(net, (6, 72, 72)))