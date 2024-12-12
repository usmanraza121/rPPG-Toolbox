from curses import flash
from inspect import CO_ASYNC_GENERATOR
from telnetlib import SSPI_LOGON
import torch
#from torch import tensor
import torch.nn as nn
import sys,os
import math
import sys
# from lib.models.multi_scale_module import MobileViTAttention, ParallelPolarizedSelfAttention
sys.path.append(os.getcwd())
#sys.path.append("lib/models")
#sys.path.append("lib/utils")
#sys.path.append("/workspace/wh/projects/DaChuang")
from lib.utils import initialize_weights # orignal code
#from utilstest import initialize_weights # our testing file
# from lib.models.common2 import DepthSeperabelConv2d as Conv
# from lib.models.common2 import SPP, Bottleneck, BottleneckCSP, Focus, Concat, Detect
from lib.models.common import Conv, SPP, Bottleneck, BottleneckCSP, Focus, Concat, Detect, SharpenConv, FusedMBConv #, Yolov4Head
# from lib.models.common import*
# from lib.models.common2 import*
from lib.models.newmodelmodules import*
from torch.nn import Upsample
from lib.utils import check_anchor_order
from lib.core.evaluate import SegmentationMetric
from lib.utils.utils import time_synchronized
from .pvt import PolypPVT       #for train
# from pvt import PolypPVT     #for main run

# from STR.transformer_seg import Encoder2D, TransConfig

# class SETR_E(nn.Module):
#     def __init__(self, patch_size=(32, 32), 
#                         in_channels=3, 
#                         out_channels=1, 
#                         hidden_size=1024, 
#                         num_hidden_layers=8, 
#                         num_attention_heads=16,
#                         decode_features=[512, 256, 128, 64],
#                         sample_rate=4,):
#         super().__init__()
#         config = TransConfig(patch_size=patch_size, 
#                             in_channels=in_channels, 
#                             out_channels=out_channels, 
#                             sample_rate=sample_rate,
#                             hidden_size=hidden_size, 
#                             num_hidden_layers=num_hidden_layers, 
#                             num_attention_heads=num_attention_heads)
#         self.encoder_2d = Encoder2D(config)
   

#     def forward(self, x):
#         _, x1 = self.encoder_2d(x)
#         x2 = self.encoder_2d(x)
#         print('x1:', x1.shape)
#         print('x2:', x2[0].shape)
#         return x 

"""Our try for model redesign"""
#==============================CNN_Multi-task Perception===================================
class CNN_P(nn.Module):
    def __init__(self, in_channel= 3, inference = False):
        super(CNN_P, self).__init__()
        """Encoder part"""
        n_classes = 13
        # inference = False
        output_ch = (4 + 1 + n_classes) * 3

        self.focus1e = Focus(in_channel,32,3)  #0
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

        self.fpe = FPEBlock(512, 512, [1, 2, 4, 8])

        self.bottleneck4e = Bottleneck(512,512) # 9
        # self.bottleneck4e = FusedMBConv(512,512) # 9

        self.conv5e = Conv(512,256,1,1) #------------ Conv(512,256,1,1) #10
        self.upsample1e = Upsample (None, 2, 'nearest') # 11

        #self.concat1e = Concat(256,256)  # 12 [11,6]
        self.bottleneck5e = Bottleneck(512,256,False) #13
        self.conv6e = Conv(256,128,1,1) #------------------Conv(256,128,1,1) #14
        self.upsample2e = Upsample (None, 2, 'nearest') # 15
        #self.concat2e = Concat(128,128) # 16 [15,4]

        "Detection Head """
        self.bottleneck1dh = BottleneckCSP(256,128,1,False) #17
        self.conv1dh = Conv(128,128,3,2) #18
        #self.concat1dh = Concat(128,128) # 19 [18,14]

        self.bottleneck2dh = BottleneckCSP(256,256,1,False) #20
        self.conv2dh = Conv(256,256,3,2) #21
        #self.concat2dh = Concat(256,256) # 22 [21,10]

        self.bottleneck3dh = BottleneckCSP(512,512,1,False) #23
        self.detect = Detect(1,[[3,9,5,11,4,20],[7,18,6,39,12,31],[19,50,38,81,68,157]],[128,256,512]) #24  Detection head[17,20,23]
        # self.detect = Detect() #([[3,9,5,11,4,20],[7,18,6,39,12,31],[19,50,38,81,68,157]],[128,256,512])
        # self.detect = Detect([256, 256, 256])

        ##self.head = Yolov4Head(output_ch, n_classes, inference)============

        """Driving Area Segmentation"""
        self.conv1das = Conv(256,128,3,1) #25..input from 16
        self.upsample1das = Upsample (None, 2, 'nearest') # 26
        self.bottleneck1das = BottleneckCSP(128,64,1,False) #27
        self.conv2das = Conv(64,32,3,1) #28
        self.upsample2das = Upsample (None, 2, 'nearest') # 29

        self.conv3das = Conv(32,16,3,1) #30
        self.bottleneck2das = BottleneckCSP(16,8,1,False) #31
        self.upsample3das = Upsample (None, 2, 'nearest') # 32
        self.conv4das = Conv(8, 2, 3, 1) #33  driv_seg==========================

        """Line Segmentation"""
        self.conv1ls = Conv(256,128,3,1) #34..input from 16
        self.upsample1ls = Upsample (None, 2, 'nearest') # 35
        self.bottleneck1ls = BottleneckCSP(128,64,1,False) #36
        self.conv2ls = Conv(64,32,3,1) #37
        self.upsample2ls = Upsample (None, 2, 'nearest') # 38

        self.conv3ls = Conv(32,16,3,1) #39
        self.bottleneck2ls = BottleneckCSP(16,8,1,False) #40
        self.upsample3ls = Upsample (None, 2, 'nearest') # 41
        self.conv4ls = Conv(8, 2, 3, 1) #42  laneline seg========================
#-------------------transformer encoder-------------------------------------
        self.pvt = PolypPVT()
        self.conv = torch.nn.Conv2d(320, 256, 1)

        # =====================detector stride============================
        Detector = self.detect # detector
        if isinstance(Detector, Detect):
            s = 128  # 2x min stride
            # for x in self.forward(torch.zeros(1, 3, s, s)):
            #     print (x.shape)
            with torch.no_grad():
                model_out = self.forward(torch.zeros(1, 3, s, s))
                detects, _, _= model_out
                Detector.stride = torch.tensor([s / x.shape[-2] for x in detects])  # forward
            # print("stride"+str(Detector.stride ))
            Detector.anchors /= Detector.stride.view(-1, 1, 1)  # Set the anchors for the corresponding scale
            check_anchor_order(Detector)
            self.stride = Detector.stride
            self._initialize_biases()
        #     -----------------------------------------66666-----------------------------
        # if isinstance(OURYOLOP, Detect):
        #     s = 128  # 2x min stride
        #     # for x in self.forward(torch.zeros(1, 3, s, s)):
        #     #     print (x.shape)
        #     with torch.no_grad():
        #         model_out = self.forward(torch.zeros(1, 3, s, s))
        #         detects, _, _= model_out
        #         Detect.stride = torch.tensor([s / x.shape[-2] for x in detects])  # forward
        #     # print("stride"+str(Detector.stride ))
        #     Detect.anchors /= Detect.stride.view(-1, 1, 1)  # Set the anchors for the corresponding scale
        #     check_anchor_order(Detect)
        #     self.stride = Detect.stride
        #     self._initialize_biases()


    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        # m = self.model[-1]  # Detect() module
        m = self.detect  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, x):
        # -----------------PVT_Encoder-----------------------
        # pvt = self.pvt(x)
        
        # x1 = pvt[0]
        # x2 = pvt[1]
        # x3 = pvt[2]
        # x3 = self.conv(x3)
        # x4 = pvt[3]


        # =====================CNN_Encoder part============================
        
        focuss1e = self.focus1e(x)                             # 0      3->32

        conv1e = self.conv1e(focuss1e)                         # 1      32->64

        bottleneck1e =self.bottleneck1e(conv1e)                # 2
      
        
        conv2e =self.conv2e(bottleneck1e)                      # 3      64->128
        
        cnn_encoder_1 =self.bottleneck2e(conv2e)               # 4      128->128     CNN_Encoder-1 [128]
      
        conv3e =self.conv3e(cnn_encoder_1)                     # 5     128->256
      
        cnn_encoder_2 =self.bottleneck3e(conv3e)               # 6     256->256      CNN_Encoder-2 [256]
        
        cnn_encoder_3 =self.conv4e(cnn_encoder_2)              # 7     256->512      CNN_Encoder-3 [512]

        # ================PVT+CNN===========================
        # encoder_1 = torch.add(cnn_encoder_1, x2)
        # encoder_2 = torch.add(cnn_encoder_2, x3)
        # encoder_3 = torch.add(cnn_encoder_3, x4)
# ---------------------------FPE part (feature Network)-------------------------

        feature_n = self.fpe(cnn_encoder_3)                   # 8     512->512     CNN_Encoder_3 To feature_network
        # feature_n = self.fpe(x4)                              #                    PVT_Encoder-3 To feature_network
        # feature_n = x4
        # feature_n = self.fpe(encoder_3)                         #                    CNN+PVT
#-----------------------------------Neck-----------------------------------------

        bottleneck4e =self.bottleneck4e(feature_n)              # 9     512->512     Feature cat Neck-1
        
        conv5e =self.conv5e(bottleneck4e)                       # 10    512->256     -->To Detect Head-3
      
        upsample1e =self.upsample1e(conv5e)                     # 11    256->upsample
      
# 
        concat1e =torch.cat((upsample1e, cnn_encoder_2),1)    # 12    256 + 256    CNN_Encoder-2 cat Neck-2          
        # concat1e =torch.cat((upsample1e,x3),1)                  #                  PVT_Encoder-2 cat Neck-2
        # concat1e =torch.cat((upsample1e,encoder_2),1)           #                    CNN+PVT

        bottleneck5e =self.bottleneck5e(concat1e)               # 13    512->256
    
        conv6e =self.conv6e(bottleneck5e)                       # 14    256->128
    
        upsample2e=self.upsample2e(conv6e)                      # 15    128->128
       
        concat2e =torch.cat((upsample2e, cnn_encoder_1),1)    # 16    128 + 128     CNN_Encoder-1 cat Neck-3 
        # concat2e =torch.cat((upsample2e,x2),1)                  #                   PVT_Encoder-2 cat Neck-2
        # concat2e =torch.cat((upsample2e,encoder_1),1)           #                     CNN+PVT

        # ==============================================Encoder end==================================

#----------------------------------Detection Head-------------------------------       
        
        bottleneck1dh=self.bottleneck1dh(concat2e)              # 17    256->128
      
        conv1dh =self.conv1dh(bottleneck1dh)                    # 18    128->128 -----to anchor_1
    
        concat1dh =torch.cat((conv1dh,conv6e),1)                # 19    128, 128 -----detection 1+2       
     
        bottleneck2dh=self.bottleneck2dh(concat1dh)             # 20    256->256 -----to anchor_2
      
        conv2dh=self.conv2dh(bottleneck2dh)                     # 21    256->256 ----3x3 conv for detection head_3
       
        concat2dh=torch.cat((conv2dh,conv5e),1)                 # 22    256 + 256 ----for detection head_3 ()
       
        bottleneck3dh=self.bottleneck3dh(concat2dh)             # 23    512, 512 ----- to anchor

        detecthead=self.detect([bottleneck1dh,bottleneck2dh,bottleneck3dh])

        # print("detecthead:", detecthead.shape)

#--------------------------------"""Driving Area Segmentation"""-------------------------------
        
        conv1das=self.conv1das(concat2e)                        # 25 256->128
      
        upsample1das=self.upsample1das(conv1das)                # 26 128->128
     
        bottleneck1das=self.bottleneck1das(upsample1das)        # 27 128->64
      
        conv2das=self.conv2das(bottleneck1das)                  # 28 64->32
       
        upsample2das=self.upsample2das(conv2das)                # 29 Upsample (2) # 29
      
#------------------------------------------------------------------
        conv3das=self.conv3das(upsample2das)                    # 30 32->16
      
        bottleneck2das=self.bottleneck2das(conv3das)            # 31 16->8
        
        upsample3das=self.upsample3das(bottleneck2das)          # 32 Upsample (2)
       
        conv4das =self.conv4das(upsample3das)                   # 33 8->2 -----drivable area segmentation out-----
      

#-------------------------------------"""Line Segmentation"""--------------------------------
        
        conv1ls =self.conv1ls(concat2e)                         # 34 256->128 from 16
    
        upsample1ls=self.upsample1ls(conv1ls)                   # 35 Upsample (2)
       
        bottleneck1ls=self.bottleneck1ls(upsample1ls)           # 36 128->64
   
        conv2ls =self.conv2ls(bottleneck1ls)                    # 37 64->32
        
        upsample2ls=self.upsample2ls(conv2ls)                   # 38 Upsample (2)
      
#------------------------------------------------------------------------
        conv3ls =self.conv3ls(upsample2ls)                      # 39 32->16
       
        bottleneck2ls=self.bottleneck2ls(conv3ls)               # 40 16->8
      
        upsample3ls=self.upsample3ls(bottleneck2ls)             # 41 Upsample (2)
      
        conv4ls =self.conv4ls(upsample3ls)                      # 42 8->classes
        
        lane_line_seg = conv4ls                                 # lane line segmentation
        driv_area_seg = conv4das                                # drivable area segmnentaion

        return detecthead, driv_area_seg, lane_line_seg


#=============================PVT_Multi-task Perception====================================
class PVT_P(nn.Module):
    def __init__(self, in_channel= 3):
        super(PVT_P, self).__init__()
        """Encoder part"""
        n_classes = 13
        # inference = False
        output_ch = (4 + 1 + n_classes) * 3

        self.focus1e = Focus(in_channel,32,3)  #0
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

        self.fpe = FPEBlock(1024, 1024, [1, 2, 4, 8])

        self.bottleneck4e = Bottleneck(1024,512) # 9
        # self.bottleneck4e = FusedMBConv(512,512) # 9

        self.conv5e = Conv(512,256,1,1) #------------ Conv(512,256,1,1) #10
        self.upsample1e = Upsample (None, 2, 'nearest') # 11

        #self.concat1e = Concat(256,256)  # 12 [11,6]
        self.bottleneck5e = Bottleneck(1024,512,False) #13
        self.conv6e = Conv(512,128,1,1) #------------------Conv(256,128,1,1) #14
        self.upsample2e = Upsample (None, 2, 'nearest') # 15
        #self.concat2e = Concat(128,128) # 16 [15,4]

        "Detection Head """
        self.bottleneck1dh = BottleneckCSP(384,128,1,False) #17
        self.conv1dh = Conv(128,128,3,2) #18
        #self.concat1dh = Concat(128,128) # 19 [18,14]

        self.bottleneck2dh = BottleneckCSP(256,256,1,False) #20
        self.conv2dh = Conv(256,256,3,2) #21
        #self.concat2dh = Concat(256,256) # 22 [21,10]

        self.bottleneck3dh = BottleneckCSP(512,512,1,False) #23
        self.detect = Detect(1,[[3,9,5,11,4,20],[7,18,6,39,12,31],[19,50,38,81,68,157]],[128,256,512]) #24  Detection head[17,20,23]
        # self.detect = Detect() #([[3,9,5,11,4,20],[7,18,6,39,12,31],[19,50,38,81,68,157]],[128,256,512])
        # self.detect = Detect([256, 256, 256])

        ##self.head = Yolov4Head(output_ch, n_classes, inference)============

        """Driving Area Segmentation"""
        self.conv1das = Conv(384,256,3,1) #25..input from 16
        self.upsample1das = Upsample (None, 2, 'nearest') # 26
        self.bottleneck1das = BottleneckCSP(256,64,1,False) #27

        self.conv2das = Conv(64,32,3,1) #28
        self.upsample2das = Upsample (None, 2, 'nearest') # 29

        self.conv3das = Conv(32,16,3,1) #30
        self.bottleneck2das = BottleneckCSP(16,8,1,False) #31
        self.upsample3das = Upsample (None, 2, 'nearest') # 32
        self.conv4das = Conv(8, 2, 3, 1) #33  driv_seg==========================

        """Line Segmentation"""
        self.conv1ls = Conv(384,256,3,1) #34..input from 16
        self.upsample1ls = Upsample (None, 2, 'nearest') # 35
        # self.bottleneck1lsnew = BottleneckCSP(256,128,1,False) #36 new add
        # self.bottleneck1ls = BottleneckCSP(128,64,1,False) #36
        self.bottleneck1ls = BottleneckCSP(256,64,1,False) #36
        self.conv2ls = Conv(64,32,3,1) #37
        self.upsample2ls = Upsample (None, 2, 'nearest') # 38

        self.conv3ls = Conv(32,16,3,1) #39
        self.bottleneck2ls = BottleneckCSP(16,8,1,False) #40
        self.upsample3ls = Upsample (None, 2, 'nearest') # 41
        self.conv4ls = Conv(8, 2, 3, 1) #42  laneline seg========================
#-------------------transformer encoder-------------------------------------
        self.pvt = PolypPVT()
        self.conv = torch.nn.Conv2d(320, 256, 1)

        # =====================detector stride============================
        Detector = self.detect # detector
        if isinstance(Detector, Detect):
            s = 128  # 2x min stride
            # for x in self.forward(torch.zeros(1, 3, s, s)):
            #     print (x.shape)
            with torch.no_grad():
                model_out = self.forward(torch.zeros(1, 3, s, s))
                detects, _, _= model_out
                Detector.stride = torch.tensor([s / x.shape[-2] for x in detects])  # forward
            # print("stride"+str(Detector.stride ))
            Detector.anchors /= Detector.stride.view(-1, 1, 1)  # Set the anchors for the corresponding scale
            check_anchor_order(Detector)
            self.stride = Detector.stride
            self._initialize_biases()

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        # m = self.model[-1]  # Detect() module
        m = self.detect  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, x):
        # -----------------PVT_Encoder-----------------------
        pvt = self.pvt(x)
        
        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x3 = self.conv(x3)
        x4 = pvt[3]


        # =====================CNN_Encoder part============================
        
        focuss1e = self.focus1e(x)                             # 0      3->32

        conv1e = self.conv1e(focuss1e)                         # 1      32->64

        bottleneck1e =self.bottleneck1e(conv1e)                # 2
      
        
        conv2e =self.conv2e(bottleneck1e)                      # 3      64->128
        
        cnn_encoder_1 =self.bottleneck2e(conv2e)               # 4      128->128     CNN_Encoder-1 [128]
      
        conv3e =self.conv3e(cnn_encoder_1)                     # 5     128->256
      
        cnn_encoder_2 =self.bottleneck3e(conv3e)               # 6     256->256      CNN_Encoder-2 [256]
        
        cnn_encoder_3 =self.conv4e(cnn_encoder_2)              # 7     256->512      CNN_Encoder-3 [512]

        # ================PVT+CNN===========================
        encoder_1 = torch.cat((cnn_encoder_1, x2), 1)
        encoder_2 = torch.cat((cnn_encoder_2, x3), 1)
        encoder_3 = torch.cat((cnn_encoder_3, x4), 1)

        # print("encoder1:", encoder_1.shape)
        # print("encoder2:", encoder_2.shape)
        # print("encoder3:", encoder_3.shape)
      
# ---------------------------FPE part (feature Network)-------------------------

        # feature_n = self.fpe(cnn_encoder_3)                   # 8     512->512     CNN_Encoder_3 To feature_network
        # feature_n = self.fpe(x4)                              #                    PVT_Encoder-3 To feature_network
        # feature_n = x4
        feature_n = self.fpe(encoder_3)                         #                    CNN+PVT
#-----------------------------------Neck-----------------------------------------

        bottleneck4e =self.bottleneck4e(feature_n)              # 9     1024->512     Feature cat Neck-1
        
        conv5e =self.conv5e(bottleneck4e)                       # 10    512->256     -->To Detect Head-3
      
        # upsample1e =self.upsample1e(conv5e)                     # 11    256->upsample
        upsample1e =self.upsample1e(bottleneck4e)               # 512-> upsample
# 
        # concat1e =torch.cat((upsample1e, cnn_encoder_2),1)    # 12    256 + 256    CNN_Encoder-2 cat Neck-2          
        # concat1e =torch.cat((upsample1e,x3),1)                  #                  PVT_Encoder-2 cat Neck-2
        concat1e =torch.cat((upsample1e,encoder_2),1)           #           512+512  CNN+PVT

        bottleneck5e =self.bottleneck5e(concat1e)               # 13    512->256///1024->512
    
        conv6e =self.conv6e(bottleneck5e)                       # 14    256->128///512->128
    
        upsample2e=self.upsample2e(conv6e)                      # 15    128->128
       
        # concat2e =torch.cat((upsample2e, cnn_encoder_1),1)    # 16    128 + 128     CNN_Encoder-1 cat Neck-3 
        # concat2e =torch.cat((upsample2e,x2),1)                  #                   PVT_Encoder-2 cat Neck-2
        concat2e =torch.cat((upsample2e,encoder_1),1)           #          256+256    CNN+PVT

        # ==============================================Encoder end==================================

#----------------------------------Detection Head-------------------------------       
        
        bottleneck1dh=self.bottleneck1dh(concat2e)              # 17    256->128
      
        conv1dh =self.conv1dh(bottleneck1dh)                    # 18    128->128 -----to anchor_1
    
        concat1dh =torch.cat((conv1dh,conv6e),1)                # 19    128, 128 -----detection 1+2       
     
        bottleneck2dh=self.bottleneck2dh(concat1dh)             # 20    256->256 -----to anchor_2
      
        conv2dh=self.conv2dh(bottleneck2dh)                     # 21    256->256 ----3x3 conv for detection head_3
       
        concat2dh=torch.cat((conv2dh,conv5e),1)                 # 22    256 + 256 ----for detection head_3 ()
       
        bottleneck3dh=self.bottleneck3dh(concat2dh)             # 23    512, 512 ----- to anchor

        detecthead=self.detect([bottleneck1dh,bottleneck2dh,bottleneck3dh])

        # print("detecthead:", detecthead.shape)

#--------------------------------"""Driving Area Segmentation"""-------------------------------
        
        conv1das=self.conv1das(concat2e)                        # 25 256->128///384->256
        # print("conv1das:", conv1das.shape)
        # convadd = torch.add(conv1das, encoder_1)  # new add
        # print("convadd:", convadd.shape)
        upsample1das=self.upsample1das(conv1das)                # 26 128->128//256
        # print("upsample1das:", upsample1das.shape)
        # print("Encoder-1:", encoder_1.shape)
        bottleneck1das=self.bottleneck1das(upsample1das)        # 27 128->64///256->64
      
        conv2das=self.conv2das(bottleneck1das)                  # 28 64->32
       
        upsample2das=self.upsample2das(conv2das)                # 29 Upsample (2) # 29
      
#------------------------------------------------------------------
        conv3das=self.conv3das(upsample2das)                    # 30 32->16
      
        bottleneck2das=self.bottleneck2das(conv3das)            # 31 16->8
        
        upsample3das=self.upsample3das(bottleneck2das)          # 32 Upsample (2)
       
        conv4das =self.conv4das(upsample3das)                   # 33 8->2 -----drivable area segmentation out-----
      

#-------------------------------------"""Line Segmentation"""--------------------------------
        
        conv1ls =self.conv1ls(concat2e)                         # 34 256->128 from 16///512->256
        # conv1ls_add = torch.add(conv1ls, encoder_1) 
    
        upsample1ls=self.upsample1ls(conv1ls)                   # 35 Upsample (2)

        # upsampl_new = self.bottleneck1lsnew(upsample1ls)   # new add 256->128

       
        bottleneck1ls=self.bottleneck1ls(upsample1ls)           # 36 128->64
   
        conv2ls =self.conv2ls(bottleneck1ls)                    # 37 64->32
        
        upsample2ls=self.upsample2ls(conv2ls)                   # 38 Upsample (2)
      
#------------------------------------------------------------------------
        conv3ls =self.conv3ls(upsample2ls)                      # 39 32->16
       
        bottleneck2ls=self.bottleneck2ls(conv3ls)               # 40 16->8
      
        upsample3ls=self.upsample3ls(bottleneck2ls)             # 41 Upsample (2)
      
        conv4ls =self.conv4ls(upsample3ls)                      # 42 8->classes
        
        lane_line_seg = conv4ls                                 # lane line segmentation
        driv_area_seg = conv4das                                # drivable area segmnentaion

        return detecthead, driv_area_seg, lane_line_seg

#=====================Hybrid Model CNN+PVT==================================

class Hybrid_P(nn.Module):
    def __init__(self, in_channel= 3):
        super(Hybrid_P, self).__init__()
        """Encoder part"""
        n_classes = 13
        # inference = False
        output_ch = (4 + 1 + n_classes) * 3

        self.focus1e = Focus(in_channel,32,3)  #0
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

        self.fpe = FPEBlock(1024, 1024, [1, 2, 4, 8])

        self.bottleneck4e = Bottleneck(1024,512) # 9
        # self.bottleneck4e = FusedMBConv(512,512) # 9

        self.conv5e = Conv(512,256,1,1) #------------ Conv(512,256,1,1) #10
        self.upsample1e = Upsample (None, 2, 'nearest') # 11

        #self.concat1e = Concat(256,256)  # 12 [11,6]
        self.bottleneck5e = Bottleneck(1024,512,False) #13
        self.conv6e = Conv(512,128,1,1) #------------------Conv(256,128,1,1) #14
        self.upsample2e = Upsample (None, 2, 'nearest') # 15
        #self.concat2e = Concat(128,128) # 16 [15,4]

        "Detection Head """
        self.bottleneck1dh = BottleneckCSP(384,128,1,False) #17
        self.conv1dh = Conv(128,128,3,2) #18
        #self.concat1dh = Concat(128,128) # 19 [18,14]

        self.bottleneck2dh = BottleneckCSP(256,256,1,False) #20
        self.conv2dh = Conv(256,256,3,2) #21
        #self.concat2dh = Concat(256,256) # 22 [21,10]

        self.bottleneck3dh = BottleneckCSP(512,512,1,False) #23
        self.detect = Detect(1,[[3,9,5,11,4,20],[7,18,6,39,12,31],[19,50,38,81,68,157]],[128,256,512]) #24  Detection head[17,20,23]
        # self.detect = Detect() #([[3,9,5,11,4,20],[7,18,6,39,12,31],[19,50,38,81,68,157]],[128,256,512])
        # self.detect = Detect([256, 256, 256])

        ##self.head = Yolov4Head(output_ch, n_classes, inference)============

        """Driving Area Segmentation"""
        self.conv1das = Conv(384,256,3,1) #25..input from 16
        self.upsample1das = Upsample (None, 2, 'nearest') # 26
        self.bottleneck1das = BottleneckCSP(256,64,1,False) #27

        self.conv2das = Conv(64,32,3,1) #28
        self.upsample2das = Upsample (None, 2, 'nearest') # 29

        self.conv3das = Conv(32,16,3,1) #30
        self.bottleneck2das = BottleneckCSP(16,8,1,False) #31
        self.upsample3das = Upsample (None, 2, 'nearest') # 32
        self.conv4das = Conv(8, 2, 3, 1) #33  driv_seg==========================

        """Line Segmentation"""
        self.conv1ls = Conv(384,256,3,1) #34..input from 16
        self.upsample1ls = Upsample (None, 2, 'nearest') # 35
        # self.bottleneck1lsnew = BottleneckCSP(256,128,1,False) #36 new add
        # self.bottleneck1ls = BottleneckCSP(128,64,1,False) #36
        self.bottleneck1ls = BottleneckCSP(256,64,1,False) #36
        self.conv2ls = Conv(64,32,3,1) #37
        self.upsample2ls = Upsample (None, 2, 'nearest') # 38

        self.conv3ls = Conv(32,16,3,1) #39
        self.bottleneck2ls = BottleneckCSP(16,8,1,False) #40
        self.upsample3ls = Upsample (None, 2, 'nearest') # 41
        self.conv4ls = Conv(8, 2, 3, 1) #42  laneline seg========================
#-------------------transformer encoder-------------------------------------
        self.pvt = PolypPVT()
        self.conv = torch.nn.Conv2d(320, 256, 1)

        # =====================detector stride============================
        Detector = self.detect # detector
        if isinstance(Detector, Detect):
            s = 128  # 2x min stride
            # for x in self.forward(torch.zeros(1, 3, s, s)):
            #     print (x.shape)
            with torch.no_grad():
                model_out = self.forward(torch.zeros(1, 3, s, s))
                detects, _, _= model_out
                Detector.stride = torch.tensor([s / x.shape[-2] for x in detects])  # forward
            # print("stride"+str(Detector.stride ))
            Detector.anchors /= Detector.stride.view(-1, 1, 1)  # Set the anchors for the corresponding scale
            check_anchor_order(Detector)
            self.stride = Detector.stride
            self._initialize_biases()

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        # m = self.model[-1]  # Detect() module
        m = self.detect  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, x):
        # -----------------PVT_Encoder-----------------------
        pvt = self.pvt(x)
        
        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x3 = self.conv(x3)
        x4 = pvt[3]


        # =====================CNN_Encoder part============================
        
        focuss1e = self.focus1e(x)                             # 0      3->32

        conv1e = self.conv1e(focuss1e)                         # 1      32->64

        bottleneck1e =self.bottleneck1e(conv1e)                # 2
      
        
        conv2e =self.conv2e(bottleneck1e)                      # 3      64->128
        
        cnn_encoder_1 =self.bottleneck2e(conv2e)               # 4      128->128     CNN_Encoder-1 [128]
      
        conv3e =self.conv3e(cnn_encoder_1)                     # 5     128->256
      
        cnn_encoder_2 =self.bottleneck3e(conv3e)               # 6     256->256      CNN_Encoder-2 [256]
        
        cnn_encoder_3 =self.conv4e(cnn_encoder_2)              # 7     256->512      CNN_Encoder-3 [512]

        # ================PVT+CNN===========================
        encoder_1 = torch.cat((cnn_encoder_1, x2), 1)
        encoder_2 = torch.cat((cnn_encoder_2, x3), 1)
        encoder_3 = torch.cat((cnn_encoder_3, x4), 1)

        # print("encoder1:", encoder_1.shape)
        # print("encoder2:", encoder_2.shape)
        # print("encoder3:", encoder_3.shape)
      
# ---------------------------FPE part (feature Network)-------------------------

        # feature_n = self.fpe(cnn_encoder_3)                   # 8     512->512     CNN_Encoder_3 To feature_network
        # feature_n = self.fpe(x4)                              #                    PVT_Encoder-3 To feature_network
        # feature_n = x4
        feature_n = self.fpe(encoder_3)                         #                    CNN+PVT
#-----------------------------------Neck-----------------------------------------

        bottleneck4e =self.bottleneck4e(feature_n)              # 9     1024->512     Feature cat Neck-1
        
        conv5e =self.conv5e(bottleneck4e)                       # 10    512->256     -->To Detect Head-3
      
        # upsample1e =self.upsample1e(conv5e)                     # 11    256->upsample
        upsample1e =self.upsample1e(bottleneck4e)               # 512-> upsample
# 
        # concat1e =torch.cat((upsample1e, cnn_encoder_2),1)    # 12    256 + 256    CNN_Encoder-2 cat Neck-2          
        # concat1e =torch.cat((upsample1e,x3),1)                  #                  PVT_Encoder-2 cat Neck-2
        concat1e =torch.cat((upsample1e,encoder_2),1)           #           512+512  CNN+PVT

        bottleneck5e =self.bottleneck5e(concat1e)               # 13    512->256///1024->512
    
        conv6e =self.conv6e(bottleneck5e)                       # 14    256->128///512->128
    
        upsample2e=self.upsample2e(conv6e)                      # 15    128->128
       
        # concat2e =torch.cat((upsample2e, cnn_encoder_1),1)    # 16    128 + 128     CNN_Encoder-1 cat Neck-3 
        # concat2e =torch.cat((upsample2e,x2),1)                  #                   PVT_Encoder-2 cat Neck-2
        concat2e =torch.cat((upsample2e,encoder_1),1)           #          256+256    CNN+PVT

        # ==============================================Encoder end==================================

#----------------------------------Detection Head-------------------------------       
        
        bottleneck1dh=self.bottleneck1dh(concat2e)              # 17    256->128
      
        conv1dh =self.conv1dh(bottleneck1dh)                    # 18    128->128 -----to anchor_1
    
        concat1dh =torch.cat((conv1dh,conv6e),1)                # 19    128, 128 -----detection 1+2       
     
        bottleneck2dh=self.bottleneck2dh(concat1dh)             # 20    256->256 -----to anchor_2
      
        conv2dh=self.conv2dh(bottleneck2dh)                     # 21    256->256 ----3x3 conv for detection head_3
       
        concat2dh=torch.cat((conv2dh,conv5e),1)                 # 22    256 + 256 ----for detection head_3 ()
       
        bottleneck3dh=self.bottleneck3dh(concat2dh)             # 23    512, 512 ----- to anchor

        detecthead=self.detect([bottleneck1dh,bottleneck2dh,bottleneck3dh])

        # print("detecthead:", detecthead.shape)

#--------------------------------"""Driving Area Segmentation"""-------------------------------
        
        conv1das=self.conv1das(concat2e)                        # 25 256->128///384->256
        # print("conv1das:", conv1das.shape)
        # convadd = torch.add(conv1das, encoder_1)  # new add
        # print("convadd:", convadd.shape)
        upsample1das=self.upsample1das(conv1das)                # 26 128->128//256
        # print("upsample1das:", upsample1das.shape)
        # print("Encoder-1:", encoder_1.shape)
        bottleneck1das=self.bottleneck1das(upsample1das)        # 27 128->64///256->64
      
        conv2das=self.conv2das(bottleneck1das)                  # 28 64->32
       
        upsample2das=self.upsample2das(conv2das)                # 29 Upsample (2) # 29
      
#------------------------------------------------------------------
        conv3das=self.conv3das(upsample2das)                    # 30 32->16
      
        bottleneck2das=self.bottleneck2das(conv3das)            # 31 16->8
        
        upsample3das=self.upsample3das(bottleneck2das)          # 32 Upsample (2)
       
        conv4das =self.conv4das(upsample3das)                   # 33 8->2 -----drivable area segmentation out-----
      

#-------------------------------------"""Line Segmentation"""--------------------------------
        
        conv1ls =self.conv1ls(concat2e)                         # 34 256->128 from 16///512->256
        # conv1ls_add = torch.add(conv1ls, encoder_1) 
    
        upsample1ls=self.upsample1ls(conv1ls)                   # 35 Upsample (2)

        # upsampl_new = self.bottleneck1lsnew(upsample1ls)   # new add 256->128

       
        bottleneck1ls=self.bottleneck1ls(upsample1ls)           # 36 128->64
   
        conv2ls =self.conv2ls(bottleneck1ls)                    # 37 64->32
        
        upsample2ls=self.upsample2ls(conv2ls)                   # 38 Upsample (2)
      
#------------------------------------------------------------------------
        conv3ls =self.conv3ls(upsample2ls)                      # 39 32->16
       
        bottleneck2ls=self.bottleneck2ls(conv3ls)               # 40 16->8
      
        upsample3ls=self.upsample3ls(bottleneck2ls)             # 41 Upsample (2)
      
        conv4ls =self.conv4ls(upsample3ls)                      # 42 8->classes
        
        lane_line_seg = conv4ls                                 # lane line segmentation
        driv_area_seg = conv4das                                # drivable area segmnentaion

        return detecthead, driv_area_seg, lane_line_seg

# ==============================under-experiment=================================================

"""Our try for model redesign"""
class YOLOP(nn.Module):
    def __init__(self, in_channel= 3):
        super(YOLOP, self).__init__()
        """Encoder part"""
        n_classes = 13
        # inference = False
        output_ch = (4 + 1 + n_classes) * 3

        self.focus1e = Focus(in_channel,32,3)  #0
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

        self.fpe = FPEBlock(1024, 1024, [1, 2, 4, 8])

        self.bottleneck4e = Bottleneck(1024,512) # 9
        # self.bottleneck4e = FusedMBConv(512,512) # 9

        self.conv5e = Conv(512,256,1,1) #------------ Conv(512,256,1,1) #10
        self.upsample1e = Upsample (None, 2, 'nearest') # 11

        #self.concat1e = Concat(256,256)  # 12 [11,6]
        self.bottleneck5e = Bottleneck(1024,512,False) #13
        self.conv6e = Conv(512,128,1,1) #------------------Conv(256,128,1,1) #14
        self.upsample2e = Upsample (None, 2, 'nearest') # 15
        #self.concat2e = Concat(128,128) # 16 [15,4]

        "Detection Head """
        self.bottleneck1dh = BottleneckCSP(384,128,1,False) #17
        self.conv1dh = Conv(128,128,3,2) #18
        #self.concat1dh = Concat(128,128) # 19 [18,14]

        self.bottleneck2dh = BottleneckCSP(256,256,1,False) #20
        self.conv2dh = Conv(256,256,3,2) #21
        #self.concat2dh = Concat(256,256) # 22 [21,10]

        self.bottleneck3dh = BottleneckCSP(512,512,1,False) #23
        self.detect = Detect(1,[[3,9,5,11,4,20],[7,18,6,39,12,31],[19,50,38,81,68,157]],[128,256,512]) #24  Detection head[17,20,23]
        # self.detect = Detect() #([[3,9,5,11,4,20],[7,18,6,39,12,31],[19,50,38,81,68,157]],[128,256,512])
        # self.detect = Detect([256, 256, 256])

        ##self.head = Yolov4Head(output_ch, n_classes, inference)============

        """Driving Area Segmentation"""
        self.conv1das = Conv(384,256,3,1) #25..input from 16
        self.upsample1das = Upsample (None, 2, 'nearest') # 26
        self.bottleneck1das = BottleneckCSP(256,64,1,False) #27

        self.conv2das = Conv(64,32,3,1) #28
        self.upsample2das = Upsample (None, 2, 'nearest') # 29

        self.conv3das = Conv(32,16,3,1) #30
        self.bottleneck2das = BottleneckCSP(16,8,1,False) #31
        self.upsample3das = Upsample (None, 2, 'nearest') # 32
        self.conv4das = Conv(8, 2, 3, 1) #33  driv_seg==========================

        """Line Segmentation"""
        self.conv1ls = Conv(384,256,3,1) #34..input from 16
        self.upsample1ls = Upsample (None, 2, 'nearest') # 35
        self.bottleneck1lsnew = BottleneckCSP(256,128,1,False) #36 new add
        self.bottleneck1ls = BottleneckCSP(128,64,1,False) #36
        # self.bottleneck1ls = BottleneckCSP(256,64,1,False) #36
        self.conv2ls = Conv(64,32,3,1) #37
        self.upsample2ls = Upsample (None, 2, 'nearest') # 38

        self.conv3ls = Conv(32,16,3,1) #39
        self.bottleneck2ls = BottleneckCSP(16,8,1,False) #40
        self.upsample3ls = Upsample (None, 2, 'nearest') # 41
        self.conv4ls = Conv(8, 2, 3, 1) #42  laneline seg========================
#-------------------transformer encoder-------------------------------------
        self.pvt = PolypPVT()
        self.conv = torch.nn.Conv2d(320, 256, 1)

        # =====================detector stride============================
        Detector = self.detect # detector
        if isinstance(Detector, Detect):
            s = 128  # 2x min stride
            # for x in self.forward(torch.zeros(1, 3, s, s)):
            #     print (x.shape)
            with torch.no_grad():
                model_out = self.forward(torch.zeros(1, 3, s, s))
                detects, _, _= model_out
                Detector.stride = torch.tensor([s / x.shape[-2] for x in detects])  # forward
            # print("stride"+str(Detector.stride ))
            Detector.anchors /= Detector.stride.view(-1, 1, 1)  # Set the anchors for the corresponding scale
            check_anchor_order(Detector)
            self.stride = Detector.stride
            self._initialize_biases()

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        # m = self.model[-1]  # Detect() module
        m = self.detect  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, x):
        # -----------------PVT_Encoder-----------------------
        pvt = self.pvt(x)
        
        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x3 = self.conv(x3)
        x4 = pvt[3]


        # =====================CNN_Encoder part============================
        
        focuss1e = self.focus1e(x)                             # 0      3->32

        conv1e = self.conv1e(focuss1e)                         # 1      32->64

        bottleneck1e =self.bottleneck1e(conv1e)                # 2
      
        
        conv2e =self.conv2e(bottleneck1e)                      # 3      64->128
        
        cnn_encoder_1 =self.bottleneck2e(conv2e)               # 4      128->128     CNN_Encoder-1 [128]
      
        conv3e =self.conv3e(cnn_encoder_1)                     # 5     128->256
      
        cnn_encoder_2 =self.bottleneck3e(conv3e)               # 6     256->256      CNN_Encoder-2 [256]
        
        cnn_encoder_3 =self.conv4e(cnn_encoder_2)              # 7     256->512      CNN_Encoder-3 [512]

        # ================PVT+CNN===========================
        encoder_1 = torch.cat((cnn_encoder_1, x2), 1)
        encoder_2 = torch.cat((cnn_encoder_2, x3), 1)
        encoder_3 = torch.cat((cnn_encoder_3, x4), 1)

        # print("encoder1:", encoder_1.shape)
        # print("encoder2:", encoder_2.shape)
        # print("encoder3:", encoder_3.shape)
      
# ---------------------------FPE part (feature Network)-------------------------

        # feature_n = self.fpe(cnn_encoder_3)                   # 8     512->512     CNN_Encoder_3 To feature_network
        # feature_n = self.fpe(x4)                              #                    PVT_Encoder-3 To feature_network
        # feature_n = x4
        feature_n = self.fpe(encoder_3)                         #                    CNN+PVT
#-----------------------------------Neck-----------------------------------------

        bottleneck4e =self.bottleneck4e(feature_n)              # 9     1024->512     Feature cat Neck-1
        
        conv5e =self.conv5e(bottleneck4e)                       # 10    512->256     -->To Detect Head-3
      
        # upsample1e =self.upsample1e(conv5e)                     # 11    256->upsample
        upsample1e =self.upsample1e(bottleneck4e)               # 512-> upsample
# 
        # concat1e =torch.cat((upsample1e, cnn_encoder_2),1)    # 12    256 + 256    CNN_Encoder-2 cat Neck-2          
        # concat1e =torch.cat((upsample1e,x3),1)                  #                  PVT_Encoder-2 cat Neck-2
        concat1e =torch.cat((upsample1e,encoder_2),1)           #           512+512  CNN+PVT

        bottleneck5e =self.bottleneck5e(concat1e)               # 13    512->256///1024->512
    
        conv6e =self.conv6e(bottleneck5e)                       # 14    256->128///512->128
    
        upsample2e=self.upsample2e(conv6e)                      # 15    128->128
       
        # concat2e =torch.cat((upsample2e, cnn_encoder_1),1)    # 16    128 + 128     CNN_Encoder-1 cat Neck-3 
        # concat2e =torch.cat((upsample2e,x2),1)                  #                   PVT_Encoder-2 cat Neck-2
        concat2e =torch.cat((upsample2e,encoder_1),1)           #          256+256    CNN+PVT

        # ==============================================Encoder end==================================

#----------------------------------Detection Head-------------------------------       
        
        bottleneck1dh=self.bottleneck1dh(concat2e)              # 17    256->128
      
        conv1dh =self.conv1dh(bottleneck1dh)                    # 18    128->128 -----to anchor_1
    
        concat1dh =torch.cat((conv1dh,conv6e),1)                # 19    128, 128 -----detection 1+2       
     
        bottleneck2dh=self.bottleneck2dh(concat1dh)             # 20    256->256 -----to anchor_2
      
        conv2dh=self.conv2dh(bottleneck2dh)                     # 21    256->256 ----3x3 conv for detection head_3
       
        concat2dh=torch.cat((conv2dh,conv5e),1)                 # 22    256 + 256 ----for detection head_3 ()
       
        bottleneck3dh=self.bottleneck3dh(concat2dh)             # 23    512, 512 ----- to anchor

        detecthead=self.detect([bottleneck1dh,bottleneck2dh,bottleneck3dh])

        # print("detecthead:", detecthead.shape)

#--------------------------------"""Driving Area Segmentation"""-------------------------------
        
        conv1das=self.conv1das(concat2e)                        # 25 256->128///384->256
        # print("conv1das:", conv1das.shape)
        convadd = torch.add(conv1das, encoder_1)  # new add
        # print("convadd:", convadd.shape)
        upsample1das=self.upsample1das(convadd)                # 26 128->128//256
        # print("upsample1das:", upsample1das.shape)
        # print("Encoder-1:", encoder_1.shape)
        bottleneck1das=self.bottleneck1das(upsample1das)        # 27 128->64///256->64
      
        conv2das=self.conv2das(bottleneck1das)                  # 28 64->32
       
        upsample2das=self.upsample2das(conv2das)                # 29 Upsample (2) # 29
      
#------------------------------------------------------------------
        conv3das=self.conv3das(upsample2das)                    # 30 32->16
      
        bottleneck2das=self.bottleneck2das(conv3das)            # 31 16->8
        
        upsample3das=self.upsample3das(bottleneck2das)          # 32 Upsample (2)
       
        conv4das =self.conv4das(upsample3das)                   # 33 8->2 -----drivable area segmentation out-----
      

#-------------------------------------"""Line Segmentation"""--------------------------------
        
        conv1ls =self.conv1ls(concat2e)                         # 34 256->128 from 16///512->256
        conv1ls_add = torch.add(conv1ls, encoder_1) 
    
        upsample1ls=self.upsample1ls(conv1ls_add)                   # 35 Upsample (2)

        upsampl_new = self.bottleneck1lsnew(upsample1ls)   # new add 256->128

       
        bottleneck1ls=self.bottleneck1ls(upsampl_new)           # 36 128->64
   
        conv2ls =self.conv2ls(bottleneck1ls)                    # 37 64->32
        
        upsample2ls=self.upsample2ls(conv2ls)                   # 38 Upsample (2)
      
#------------------------------------------------------------------------
        conv3ls =self.conv3ls(upsample2ls)                      # 39 32->16
       
        bottleneck2ls=self.bottleneck2ls(conv3ls)               # 40 16->8
      
        upsample3ls=self.upsample3ls(bottleneck2ls)             # 41 Upsample (2)
      
        conv4ls =self.conv4ls(upsample3ls)                      # 42 8->classes
        
        lane_line_seg = conv4ls                                 # lane line segmentation
        driv_area_seg = conv4das                                # drivable area segmnentaion

        return detecthead, driv_area_seg, lane_line_seg

if __name__ == "__main__":
    from torch.utils.tensorboard import SummaryWriter
    #model = get_net(False)
    model =YOLOP(3)
    input_ = torch.randn((1, 3, 640, 640))
#     print('input;;', input_.type)
    gt_ = torch.rand((1, 2, 256, 256))
    metric = SegmentationMetric(2)
    
#     model_out,SAD_out, lout = model(input_)
#     model_out,SAD_out, lout = model(input_)
#     detects, dring_area_seg, lane_line_seg = model_out

    detects, dring_area_seg, lane_line_seg = model(input_)
    # Da_fmap, LL_fmap = SAD_out
    for det in detects:
        print('det:', det.shape)
    print(dring_area_seg.shape)
    print(lane_line_seg.shape)

 
# if __name__ == "__main__":
#     net = SETR_E(patch_size=(32, 32), 
#                     in_channels=3, 
#                     out_channels=1, 
#                     hidden_size=1024, 
#                     sample_rate=5,
#                     num_hidden_layers=1, 
#                     num_attention_heads=16, 
#                     decode_features=[512, 256, 128, 64])
    
#     t1 = torch.rand(1, 3, 512, 512)
#     x2 = net(t1)
#     print("input: " + str(x2.shape))