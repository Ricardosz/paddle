#import torch
#import torch.nn as nn
#from torch.nn.parameter import Parameter


import paddle.fluid as fluid
import numpy as np
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph import Conv2D, Pool2D, BatchNorm, Linear,InstanceNorm,PRelu,SpectralNorm
from paddle.fluid.dygraph import Sequential
import paddle.fluid.dygraph.nn as nn
#from paddle.fluid.Tensor import tensor

class ResnetGenerator(fluid.dygraph.Layer):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, img_size=256, light=False):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.img_size = img_size
        self.light = light

      
        DownBlock = []
        DownBlock += [
                      #fluid.layers.pad2d(3),
                      Conv2D(num_channels=input_nc, num_filters=ngf, filter_size=7, stride=1, padding=3, bias_attr=False),
                      InstanceNorm(ngf),
                      PRelu(mode="all")
                      #BatchNorm(ngf,act='relu')
                      #fluid.layers.instance_norm(ngf)
                      #nn.ReLU(True)
                      ]
        self.conv1=Conv2D(input_nc, ngf, 7)
        self.instance_norm=InstanceNorm(ngf)
        #self.n_downsampling=n_downsampling
        # Down-Sampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            DownBlock += [
                          #fluid.layers.pad2d(1),
                          Conv2D(ngf * mult, ngf * mult * 2, filter_size=3, stride=2, padding=1, bias_attr=False),
                          InstanceNorm(ngf * mult * 2),
                          PRelu(mode="all")                          
                          #BatchNorm(ngf * mult * 2,act='relu')
                          #fluid.layers.instance_norm(ngf * mult * 2)
                          #nn.ReLU(True)
                          ]
        mult = 2**0
        self.conv2=Conv2D(ngf * mult, ngf * mult * 2, filter_size=3, stride=2, padding=0, bias_attr=False)
        self.instance_norm1=InstanceNorm(ngf * mult * 2)
        mult = 2**1
        self.conv3=Conv2D(ngf * mult, ngf * mult * 2, filter_size=3, stride=2, padding=0, bias_attr=False)

        # Down-Sampling Bottleneck
        mult = 2**n_downsampling
        for i in range(n_blocks):
            DownBlock += [ResnetBlock(ngf * mult, use_bias=False)]

        self.renetblock=ResnetBlock(ngf * mult, use_bias=False)
        # Class Activation Map
        self.gap_fc = Linear(ngf * mult, 1, bias_attr=False)
        self.gmp_fc = Linear(ngf * mult, 1, bias_attr=False)
        self.conv1x1 = Conv2D(ngf * mult * 2, ngf * mult, filter_size=1, stride=1, bias_attr=True,act='relu')
        #self.relu = nn.ReLU(True)

        # Gamma, Beta block
        if self.light:
            FC = [Linear(ngf * mult, ngf * mult, bias_attr=False,act='relu'),
                  #nn.ReLU(True),
                  Linear(ngf * mult, ngf * mult, bias_attr=False,act='relu')
                  #nn.ReLU(True)
                  ]
        else:
            FC = [Linear(img_size // mult * img_size // mult * ngf * mult, ngf * mult, bias_attr=False,act='relu'),
                  #nn.ReLU(True),
                  Linear(ngf * mult, ngf * mult, bias_attr=False,act='relu')
                  #nn.ReLU(True)
                  ]
        self.gamma = Linear(ngf * mult, ngf * mult, bias_attr=False)
        self.beta = Linear(ngf * mult, ngf * mult, bias_attr=False)

        # Up-Sampling Bottleneck
        for i in range(n_blocks):
            setattr(self, 'UpBlock1_' + str(i+1), ResnetAdaILNBlock(ngf * mult, use_bias=False))

        # Up-Sampling
        UpBlock2 = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            UpBlock2 += [#nn.Upsample(scale_factor=2, mode='nearest'),
                         #fluid.layers.pad2d(1),
                         Upsample(),
                         Conv2D(ngf * mult, int(ngf * mult / 2), filter_size=3, stride=1, padding=1, bias_attr=False,act='relu'),
                         ILN(int(ngf * mult / 2))
                         #nn.ReLU(True)
                         ]

        UpBlock2 += [
                     #fluid.layers.pad2d(3),
                     Conv2D(ngf, output_nc, filter_size=7, stride=1, padding=3, bias_attr=False,act='tanh')
                     #nn.Tanh()
                     ]

        self.DownBlock = Sequential(*DownBlock)
        self.FC = Sequential(*FC)
        self.UpBlock2 = Sequential(*UpBlock2)

    def forward(self, input):
        #x = fluid.layers.pad2d(input=input, paddings=[3, 3, 3, 3], mode='reflect')
        #x=self.conv1(x)
        #x=self.instance_norm(x)
        #x=fluid.layers.relu(x)
        #for i in range(self.n_downsampling):
            #mult = 2**i     
            
        x = self.DownBlock(input)
        #torch.Size([1, 256, 64, 64])
        #gap torch.Size([1, 256, 1, 1])
        
        #gap = fluid.layers.adaptive_pool2d(x, x.shape[-1],pool_type='avg')
        
        gap =Pool2D(pool_size=x.shape[-1],pool_stride=x.shape[-1],pool_type='avg')(x)
        gap=fluid.layers.reshape(gap, shape=[x.shape[0], -1]) #torch.Size([1, 1])
        gap_logit = self.gap_fc(gap)
        gap_weight = list(self.gap_fc.parameters())[0] #torch.Size([1, 256])
        gap_weight = fluid.layers.unsqueeze(input=gap_weight, axes=[0])
        gap_weight = fluid.layers.unsqueeze(input=gap_weight, axes=[3])
        gap = x * gap_weight    #torch.Size([1, 256, 64, 64])
        #gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        #gmp = fluid.layers.adaptive_pool2d(x, 1,pool_type='max')
        gmp =Pool2D(pool_size=x.shape[-1],pool_stride=x.shape[-1],pool_type='max')(x)
        gmp=fluid.layers.reshape(gmp, shape=[x.shape[0], -1])
        gmp_logit = self.gmp_fc(gmp)
        gmp_weight = list(self.gmp_fc.parameters())[0]
        
        gmp_weight = fluid.layers.unsqueeze(input=gmp_weight, axes=[0])
        gmp_weight = fluid.layers.unsqueeze(input=gmp_weight, axes=[3])     
        gmp = x * gmp_weight  #torch.Size([1, 256, 64, 64])
        #gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

        #cam_logit = torch.cat([gap_logit, gmp_logit], 1)
        #x = torch.cat([gap, gmp], 1)
        cam_logit = fluid.layers.concat([gap_logit, gmp_logit], 1)
        x = fluid.layers.concat([gap, gmp], 1)   #torch.Size([1, 512, 64, 64])      
        x = self.conv1x1(x) #torch.Size([1, 256, 64, 64])
        #x = self.relu(self.conv1x1(x))
        #torch.Size([1, 256, 64, 64])
        #heatmap = torch.sum(x, dim=1, keepdim=True)
        heatmap = fluid.layers.reduce_sum(x, dim=1, keep_dim=True)
        #heatmap torch.Size([1, 1, 64, 64])
        if self.light:
            x_ = fluid.layers.adaptive_pool2d(x, 1,pool_type='avg')
            x_=fluid.layers.reshape(x_, shape=[x.shape[0], -1])
            x_ = self.FC(x_)
        else:
            x_=fluid.layers.reshape(x, shape=[x.shape[0], -1])
            x_ = self.FC(x_)
        gamma, beta = self.gamma(x_), self.beta(x_)
        # gamma torch.Size([1, 256]) beta torch.Size([1, 256])

        for i in range(self.n_blocks):
            x = getattr(self, 'UpBlock1_' + str(i+1))(x, gamma, beta)
        out = self.UpBlock2(x)
        #out torch.Size([1, 3, 256, 256]) cam_logit torch.Size([1, 2])  heatmap torch.Size([1, 1, 64, 64])
        return out, cam_logit, heatmap


class ResnetBlock(fluid.dygraph.Layer):
    def __init__(self, dim, use_bias):
        super(ResnetBlock, self).__init__()
        conv_block = []
        conv_block += [
                       #fluid.layers.pad2d(1),
                       Conv2D(dim, dim, filter_size=3, stride=1, padding=1, bias_attr=use_bias),
                       InstanceNorm(dim),
                       PRelu(mode="all")                        
                       #BatchNorm(dim,act='relu')
                       #nn.ReLU(True)
                       ]

        conv_block += [
                       #fluid.layers.pad2d(1),
                       Conv2D(dim, dim, filter_size=3, stride=1, padding=1, bias_attr=use_bias),
                       InstanceNorm(dim)]

        self.conv_block = Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetAdaILNBlock(fluid.dygraph.Layer):
    def __init__(self, dim, use_bias):
        super(ResnetAdaILNBlock, self).__init__()
        #self.pad1 = fluid.layers.pad2d()
        self.conv1 = Conv2D(dim, dim, filter_size=3, stride=1, padding=1, bias_attr=use_bias)
        self.norm1 = adaILN(dim)
        #self.relu1 = nn.ReLU(True)

        #self.pad2 = fluid.layers.pad2d()
        self.conv2 = Conv2D(dim, dim, filter_size=3, stride=1, padding=1, bias_attr=use_bias)
        self.norm2 = adaILN(dim)

    def forward(self, x, gamma, beta):
        #out = self.pad1(x)
        out = self.conv1(x)
        out = self.norm1(out, gamma, beta)
        out=fluid.layers.relu(out)
        #out = self.relu1(out)
        #out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta)

        return out + x


class adaILN(fluid.dygraph.Layer):
    def __init__(self, num_features, eps=1e-5):
        super(adaILN, self).__init__()
        self.eps = eps
        self.rho = fluid.layers.fill_constant(shape=[1, num_features, 1, 1], value=0.9, dtype='float32')
        #self.rho = Parameter(fluid.Tensor(1, num_features, 1, 1))
        #self.rho = fluid.Tensor()
        #self.rho.set(np.ndarray([1, num_features, 1, 1]), fluid.CPUPlace())        
        #self.rho = fluid.Tensor(1, num_features, 1, 1)
        #self.rho=np.ndarray([1, num_features, 1, 1])
        #self.rho.data.fill_(0.9)

    def forward(self, input, gamma, beta):
        #torch.Size([1, 256, 64, 64])
        ninput=input.numpy()
        #self.rho=self.rho.numpy()
        in_mean=np.mean(ninput, axis=(2, 3), keepdims=True)
        in_var=np.var(ninput, axis=(2, 3), keepdims=True)
        out_in = (ninput - in_mean) / np.sqrt(in_var + self.eps)
        ln_mean, ln_var = np.mean(ninput, axis=(1, 2, 3), keepdims=True), np.var(ninput, axis=(1, 2, 3), keepdims=True)
        out_ln = (ninput - ln_mean) / np.sqrt(ln_var + self.eps)
        out_in = fluid.dygraph.base.to_variable(out_in)
        out_ln = fluid.dygraph.base.to_variable(out_ln)
        ninput = fluid.dygraph.base.to_variable(ninput)
        #out = fluid.dygraph.base.to_variable(out)
        out = self.rho * out_in + (1-self.rho) * out_ln
        
        #t = fluid.Tensor()
        #t.set(out, fluid.CPUPlace())   
        gamma = fluid.layers.unsqueeze(input=gamma, axes=[2])
        gamma = fluid.layers.unsqueeze(input=gamma, axes=[3])   
        beta = fluid.layers.unsqueeze(input=beta, axes=[2])
        beta = fluid.layers.unsqueeze(input=beta, axes=[3])  
        out = out *gamma+beta
        #in_mean, in_var = fluid.layers.reduce_mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        #out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        #ln_mean, ln_var = fluid.layers.reduce_mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        #out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        #out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1-self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        #out = out * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)
        #out torch.Size([1, 256, 64, 64])
        return out


class ILN(fluid.dygraph.Layer):
    def __init__(self, num_features, eps=1e-5):
        super(ILN, self).__init__()
        self.eps = eps
        #self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        #self.gamma = Parameter(torch.Tensor(1, num_features, 1, 1))
        #self.beta = Parameter(torch.Tensor(1, num_features, 1, 1))
        #self.rho = fluid.Tensor(1, num_features, 1, 1)
        #self.gamma = fluid.Tensor(1, num_features, 1, 1)
        #self.beta = fluid.Tensor(1, num_features, 1, 1)  
        self.rho = fluid.layers.fill_constant(shape=[1, num_features, 1, 1], value=0.0, dtype='float32')
        self.gamma = fluid.layers.fill_constant(shape=[1, num_features, 1, 1], value=1.0, dtype='float32')
        self.beta = fluid.layers.fill_constant(shape=[1, num_features, 1, 1], value=0.0, dtype='float32')
        #self.rho = fluid.Tensor()
        #self.rho.set(np.ndarray([1, num_features, 1, 1]), fluid.CPUPlace())       
        #self.gamma = fluid.Tensor()
        #self.gamma.set(np.ndarray([1, num_features, 1, 1]), fluid.CPUPlace())   
        #self.beta = fluid.Tensor()
        #self.beta.set(np.ndarray([1, num_features, 1, 1]), fluid.CPUPlace())   
        
        #self.rho.data.fill_(0.0)
        #self.gamma.data.fill_(1.0)
        #self.beta.data.fill_(0.0)

    def forward(self, input):
        #torch.Size([1, 128, 128, 128])
        ninput=input.numpy()
        #self.rho=self.rho.numpy()
        #self.gamma=self.gamma.numpy()
        #self.beta=self.beta.numpy()
        in_mean=np.mean(ninput, axis=(2, 3), keepdims=True)
        in_var=np.var(ninput, axis=(2, 3), keepdims=True)
        out_in = (ninput - in_mean) / np.sqrt(in_var + self.eps)
        ln_mean, ln_var = np.mean(ninput, axis=(1, 2, 3), keepdims=True), np.var(ninput, axis=(1, 2, 3), keepdims=True)
        out_ln = (ninput - ln_mean) / np.sqrt(ln_var + self.eps)
        out_in = fluid.dygraph.base.to_variable(out_in)
        out_ln = fluid.dygraph.base.to_variable(out_ln)
        ninput = fluid.dygraph.base.to_variable(ninput)        
        out = self.rho * out_in + (1-self.rho) * out_ln
        out = out * self.gamma + self.beta
        #out = fluid.dygraph.base.to_variable(out)
        #t = fluid.Tensor()
        #t.set(out, fluid.CPUPlace()) 
        
        #in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        #out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        #ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        #out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        #out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1-self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        #out = out * self.gamma.expand(input.shape[0], -1, -1, -1) + self.beta.expand(input.shape[0], -1, -1, -1)
        #out torch.Size([1, 128, 128, 128])
        return out


class Discriminator(fluid.dygraph.Layer):
    def __init__(self, input_nc, ndf=64, n_layers=5):
        super(Discriminator, self).__init__()
        #model = [fluid.layers.pad2d(1),
                 #nn.utils.spectral_norm(
                 #nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=0, bias=True)),
                 #nn.LeakyReLU(0.2, True)]
        model = [
                 #fluid.layers.pad2d(1),
                 Conv2D(input_nc, ndf, filter_size=4, stride=2, padding=1, bias_attr=True,act='leaky_relu')]        

        #for i in range(1, n_layers - 2):
            #mult = 2 ** (i - 1)
            #model += [fluid.layers.pad2d(1),
                      #nn.utils.spectral_norm(
                      #nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=2, padding=0, bias=True)),
                      #nn.LeakyReLU(0.2, True)]
        for i in range(1, n_layers - 2):
            mult = 2 ** (i - 1)
            model += [
                      #fluid.layers.pad2d(1),
                      Conv2D(ndf * mult, ndf * mult * 2, filter_size=4, stride=2, padding=1, bias_attr=True,act='leaky_relu')
                      ]        

        mult = 2 ** (n_layers - 2 - 1)
        #model += [fluid.layers.pad2d(1),
                  #nn.utils.spectral_norm(
                  #nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=1, padding=0, bias=True)),
                  #nn.LeakyReLU(0.2, True)]
        model += [
                  #fluid.layers.pad2d(1),
                  Conv2D(ndf * mult, ndf * mult * 2, filter_size=4, stride=1, padding=1, bias_attr=True,act='leaky_relu'),
                  ]        

        # Class Activation Map
        mult = 2 ** (n_layers - 2)
        #self.gap_fc = nn.utils.spectral_norm(nn.Linear(ndf * mult, 1, bias=False))
        #self.gmp_fc = nn.utils.spectral_norm(nn.Linear(ndf * mult, 1, bias=False))
        #self.conv1x1 = nn.Conv2d(ndf * mult * 2, ndf * mult, kernel_size=1, stride=1, bias=True)
        #self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.gap_fc = Linear(ndf * mult, 1, bias_attr=False)
        self.gmp_fc = Linear(ndf * mult, 1, bias_attr=False)
        self.conv1x1 =Conv2D(ndf * mult * 2, ndf * mult, filter_size=1, stride=1, bias_attr=True)
        #self.leaky_relu = nn.LeakyReLU(0.2, True)        

        #self.pad = fluid.layers.pad2d(1)
        #self.conv = nn.utils.spectral_norm(
            #nn.Conv2d(ndf * mult, 1, kernel_size=4, stride=1, padding=0, bias=False))
        self.conv = Conv2D(ndf * mult, 1, filter_size=4, stride=1, padding=1, bias_attr=False)    

        self.model = Sequential(*model)

    def forward(self, input):
        x = self.model(input) #[1, 2048, 2, 2]

        #gap = fluid.layers.adaptive_pool2d(x, 1,pool_type='avg')
        gap =Pool2D(pool_size=x.shape[-1],pool_stride=x.shape[-1],pool_type='avg')(x) #[1, 2048, 1, 1]
        gap=fluid.layers.reshape(gap, shape=[x.shape[0], -1]) 
        gap_logit = self.gap_fc(gap)#torch.Size([1, 1])
        gap_weight = list(self.gap_fc.parameters())[0]
        gap_weight = fluid.layers.unsqueeze(input=gap_weight, axes=[0])
        gap_weight = fluid.layers.unsqueeze(input=gap_weight, axes=[3])   
        gap = x * gap_weight #[1, 2048, 2, 2]
        #gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        #gmp = fluid.layers.adaptive_pool2d(x, 1,pool_type='max')
        gmp =Pool2D(pool_size=x.shape[-1],pool_stride=x.shape[-1],pool_type='max')(x)
        gmp=fluid.layers.reshape(gmp, shape=[x.shape[0], -1])        
        gmp_logit = self.gmp_fc(gmp)
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp_weight = fluid.layers.unsqueeze(input=gmp_weight, axes=[0])
        gmp_weight = fluid.layers.unsqueeze(input=gmp_weight, axes=[3])          
        #gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)
        gmp = x * gmp_weight
        cam_logit = fluid.layers.concat([gap_logit, gmp_logit], 1)
        x = fluid.layers.concat([gap, gmp], 1)
        x=fluid.layers.leaky_relu(self.conv1x1(x))
        #x = self.leaky_relu(self.conv1x1(x))

        heatmap = fluid.layers.reduce_sum(x, dim=1, keep_dim=True)

        #x = self.pad(x)
        out = self.conv(x)

        return out, cam_logit, heatmap

# 定义上采样模块
class Upsample(fluid.dygraph.Layer):
    def __init__(self, scale=2):
        super(Upsample, self).__init__()
        self.scale = scale

    def forward(self, inputs):
        # get dynamic upsample output shape
        shape_nchw = fluid.layers.shape(inputs)
        shape_hw = fluid.layers.slice(shape_nchw, axes=[0], starts=[2], ends=[4])
        shape_hw.stop_gradient = True
        in_shape = fluid.layers.cast(shape_hw, dtype='int32')
        out_shape = in_shape * self.scale
        out_shape.stop_gradient = True

        # reisze by actual_shape
        out = fluid.layers.resize_nearest(
            input=inputs, scale=self.scale, actual_shape=out_shape)
        return out
class RhoClipper(object):

    def __init__(self, min, max):
        self.clip_min = min
        self.clip_max = max
        assert min < max

    def __call__(self, module):

        if hasattr(module, 'rho'):
            w = module.rho.data
            w = w.clamp(self.clip_min, self.clip_max)
            module.rho.data = w


class BCEWithLogitsLoss():
    def __init__(self, weight=None, reduction='mean'):
        self.weight = weight
        self.reduction = 'mean'

    def __call__(self, x, label):
        out = fluid.layers.sigmoid_cross_entropy_with_logits(x, label)
        if self.reduction == 'sum':
            return fluid.layers.reduce_sum(out)
        elif self.reduction == 'mean':
            return fluid.layers.reduce_mean(out)
        else:
            return out