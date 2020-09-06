# UGATIT 论文复现

## 项目简介
使用百度的Paddle框架复现UGATIT 论文
## 准备资料
### 论文地址
[https://arxiv.org/abs/1907.10830](https://arxiv.org/abs/1907.10830)
### 参考代码
基于PyTorch :[https://github.com/znxlwm/UGATIT-pytorch](https://github.com/znxlwm/UGATIT-pytorch)
评价标准:[https://github.com/taki0112/GAN_Metrics-Tensorflow](https://github.com/taki0112/GAN_Metrics-Tensorflow)
### 数据集
[https://aistudio.baidu.com/aistudio/datasetdetail/48778](https://aistudio.baidu.com/aistudio/datasetdetail/48778)
### 对照API
[Paddle1.8-Pytorch-API对照表.docx](https://www.yuque.com/attachments/yuque/0/2020/docx/448161/1598798412992-94abdbba-e61e-4eb4-83e4-ebfe5c072e27.docx?_lake_card=%7B%22uid%22%3A%221598798411982-0%22%2C%22src%22%3A%22https%3A%2F%2Fwww.yuque.com%2Fattachments%2Fyuque%2F0%2F2020%2Fdocx%2F448161%2F1598798412992-94abdbba-e61e-4eb4-83e4-ebfe5c072e27.docx%22%2C%22name%22%3A%22Paddle1.8-Pytorch-API%E5%AF%B9%E7%85%A7%E8%A1%A8.docx%22%2C%22size%22%3A24340%2C%22type%22%3A%22application%2Fvnd.openxmlformats-officedocument.wordprocessingml.document%22%2C%22ext%22%3A%22docx%22%2C%22progress%22%3A%7B%22percent%22%3A99%7D%2C%22status%22%3A%22done%22%2C%22percent%22%3A0%2C%22id%22%3A%22OpPDr%22%2C%22card%22%3A%22file%22%7D)
### 测试验证脚本
[GAN_Metrics-Tensorflow](https://github.com/taki0112/GAN_Metrics-Tensorflow)
## 论文研读
### 论文实现内容

- 实现了无监督的图像翻译问题，两个图像域间纹理和图像差别很大时的 风格转换
- 实现了相同的网络结构和超参数同时进行需要保持shape的图像翻译
### GAN实现内容
![image.png](https://cdn.nlark.com/yuque/0/2020/png/448161/1598798918173-806b4250-4ce4-4ff4-b414-08b5cbf2bda6.png#align=left&display=inline&height=491&margin=%5Bobject%20Object%5D&name=image.png&originHeight=491&originWidth=1155&size=163363&status=done&style=none&width=1155)
### 创新点
**无监督图像跨域转换模型，学习两个不同域内映射图像的功能**

- 新的归一化模块和新的归一化函数AdaLIN构成的一种新的无监督图像到图像的转换方法
- 自适应的归一化函数AdaLIN,增强模型鲁棒性
- 利用attention模块（添加辅助分类器）,增强生成器的生成能力，更好的区分源域和目标域，以及判别器的判别能力，更好的区分生成图像和原始图像
### 创新方法
#### 引入新的可学习的归一化方法AdaLIN

   - Layer Norm更多的考虑输入特征通道之间 的相关性，LN比IN风格转换更彻底，但是 语义信息保存不足 
   - Instance Norm更多的考虑的单个特征通 道的内容，IN比LN更好的保存原图像的语 义信息，但是风格转换不彻底

![image.png](https://cdn.nlark.com/yuque/0/2020/png/448161/1598799185837-5928e874-cbf9-4ea3-8d7c-cb1baed221ae.png#align=left&display=inline&height=169&margin=%5Bobject%20Object%5D&name=image.png&originHeight=169&originWidth=412&size=16043&status=done&style=none&width=412)
![image.png](https://cdn.nlark.com/yuque/0/2020/png/448161/1598799193355-21e4a751-c164-4082-835b-62db0096e41d.png#align=left&display=inline&height=407&margin=%5Bobject%20Object%5D&name=image.png&originHeight=407&originWidth=628&size=130424&status=done&style=none&width=628)
#### 无监督图像跨域转换模型
通过网络的encode编码阶段得到特征图，然后通过对特征图的最大池化，经过全连接 层输出一个节点的预测，然后将这个全连接层的参数，和特征图相乘得到化attention 的特征图
**![image.png](https://cdn.nlark.com/yuque/0/2020/png/448161/1598799265792-2eee719e-f187-429c-99d8-001ecc62c871.png#align=left&display=inline&height=318&margin=%5Bobject%20Object%5D&name=image.png&originHeight=318&originWidth=1095&size=133916&status=done&style=none&width=1095)**
## 论文结果
计算真实图像和生成图像之间的平方最大均值差异(squared Maximum Mean Discrepancy)，值 越小表示真实图像与生成图像之间有更多视觉相似性，图像翻译的效果越好。
![image.png](https://cdn.nlark.com/yuque/0/2020/png/448161/1598799311740-1c6e7830-37bd-49bd-8599-f8229e7f84df.png#align=left&display=inline&height=354&margin=%5Bobject%20Object%5D&name=image.png&originHeight=354&originWidth=897&size=128031&status=done&style=none&width=897)
#### 结果展示


![image.png](https://cdn.nlark.com/yuque/0/2020/png/448161/1598799348009-79df22d7-1cd3-405f-bd50-52115fdc84ec.png#align=left&display=inline&height=533&margin=%5Bobject%20Object%5D&name=image.png&originHeight=533&originWidth=831&size=1025358&status=done&style=none&width=831)
#### 论文总结
**构建了具有自适应归一化的生成对抗网络：**

- 无监督的不同域间的图像到图像转换任务 
- 论文结构简单，代码简洁
- Attention模块如何实现 
- 归一化函数如何使用
#### 应用场景
![image.png](https://cdn.nlark.com/yuque/0/2020/png/448161/1598799494112-1ae6f10d-b970-42e4-b715-32726f339371.png#align=left&display=inline&height=543&margin=%5Bobject%20Object%5D&name=image.png&originHeight=543&originWidth=1166&size=923281&status=done&style=none&width=1166)
## 源码解读
参考PyTorch 版本的源码来理解对应的网络层构建
### 生成器
```
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, img_size=256, light=False):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc    #输入通道数 --> 3
        self.output_nc = output_nc  #输出通道数 --> 3
        self.ngf = ngf              #第一层卷积后的通道数 --> 64
        self.n_blocks = n_blocks	#残差块数 --> 6
        self.img_size = img_size    #图像size --> 256
        self.light = light          #是否使用轻量级模型

        DownBlock = []
        # 先通过一个卷积核尺寸为7的卷积层，图片大小不变，通道数变为64
        DownBlock += [nn.ReflectionPad2d(3),
                      nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=0, bias=False),
                      nn.InstanceNorm2d(ngf),
                      nn.ReLU(True)]

        # Down-Sampling --> 下采样模块
        n_downsampling = 2
        # 两层下采样，img_size缩小4倍（64），通道数扩大4倍（256）
        for i in range(n_downsampling): 
            mult = 2**i
            DownBlock += [nn.ReflectionPad2d(1),
                          nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=0, bias=False),
                          nn.InstanceNorm2d(ngf * mult * 2),
                          nn.ReLU(True)]

        # Down-Sampling Bottleneck  --> 编码器中的残差模块
        mult = 2**n_downsampling
        # 6个残差块，尺寸和通道数都不变
        for i in range(n_blocks):
            DownBlock += [ResnetBlock(ngf * mult, use_bias=False)]

        # Class Activation Map --> 产生类别激活图
        #接着global average pooling后的全连接层
        self.gap_fc = nn.Linear(ngf * mult, 1, bias=False)
        #接着global max pooling后的全连接层
        self.gmp_fc = nn.Linear(ngf * mult, 1, bias=False)
        #下面1x1卷积和激活函数，是为了得到两个pooling合并后的特征图
        self.conv1x1 = nn.Conv2d(ngf * mult * 2, ngf * mult, kernel_size=1, stride=1, bias=True)
        self.relu = nn.ReLU(True)

        # Gamma, Beta block --> 生成自适应 L-B Normalization(AdaILN)中的Gamma, Beta
        if self.light: # 确定轻量级，FC使用的是两个256 --> 256的全连接层
            FC = [nn.Linear(ngf * mult, ngf * mult, bias=False),
                  nn.ReLU(True),
                  nn.Linear(ngf * mult, ngf * mult, bias=False),
                  nn.ReLU(True)]
        else:
            #不是轻量级，则下面的1024x1024 --> 256的全连接层和一个256 --> 256的全连接层
            FC = [nn.Linear(img_size // mult * img_size // mult * ngf * mult, ngf * mult, bias=False), # (1024x1014, 64x4) crazy
                  nn.ReLU(True),
                  nn.Linear(ngf * mult, ngf * mult, bias=False),
                  nn.ReLU(True)]
        #AdaILN中的Gamma, Beta
        self.gamma = nn.Linear(ngf * mult, ngf * mult, bias=False)
        self.beta = nn.Linear(ngf * mult, ngf * mult, bias=False)
		
        # Up-Sampling Bottleneck --> 解码器中的自适应残差模块
        for i in range(n_blocks):
            setattr(self, 'UpBlock1_' + str(i+1), ResnetAdaILNBlock(ngf * mult, use_bias=False))

        # Up-Sampling --> 解码器中的上采样模块
        UpBlock2 = []
        #上采样与编码器的下采样对应
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            UpBlock2 += [nn.Upsample(scale_factor=2, mode='nearest'),
                         nn.ReflectionPad2d(1),
                         nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=0, bias=False),
                         ILN(int(ngf * mult / 2)), #注:只有自适应残差块使用AdaILN
                         nn.ReLU(True)]
		#最后一层卷积层，与最开始的卷积层对应
        UpBlock2 += [nn.ReflectionPad2d(3),
                     nn.Conv2d(ngf, output_nc, kernel_size=7, stride=1, padding=0, bias=False),
                     nn.Tanh()]
		
        self.DownBlock = nn.Sequential(*DownBlock) #编码器整个模块
        self.FC = nn.Sequential(*FC)               #生成gamma,beta的全连接层模块
        self.UpBlock2 = nn.Sequential(*UpBlock2)   #只包含上采样后的模块，不包含残差块

    def forward(self, input):
        x = self.DownBlock(input)  #得到编码器的输出,对应途中encoder feature map

        gap = torch.nn.functional.adaptive_avg_pool2d(x, 1) #全局平均池化
        gap_logit = self.gap_fc(gap.view(x.shape[0], -1)) #gap的预测
        gap_weight = list(self.gap_fc.parameters())[0] #self.gap_fc的权重参数
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3) #得到全局平均池化加持权重的特征图

        gmp = torch.nn.functional.adaptive_max_pool2d(x, 1) #全局最大池化
        gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1)) #gmp的预测
        gmp_weight = list(self.gmp_fc.parameters())[0] #self.gmp_fc的权重参数
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3) #得到全局最大池化加持权重的特征图

        cam_logit = torch.cat([gap_logit, gmp_logit], 1) #结合gap和gmp的cam_logit预测
        x = torch.cat([gap, gmp], 1)  #结合两种池化后的特征图，通道数512
        x = self.relu(self.conv1x1(x)) #接入一个卷积层，通道数512转换为256

        heatmap = torch.sum(x, dim=1, keepdim=True) #得到注意力热力图

        if self.light:
            x_ = torch.nn.functional.adaptive_avg_pool2d(x, 1) #轻量级则先经过一个gap
            x_ = self.FC(x_.view(x_.shape[0], -1))
        else:
            x_ = self.FC(x.view(x.shape[0], -1))
        gamma, beta = self.gamma(x_), self.beta(x_) #得到自适应gamma和beta


        for i in range(self.n_blocks):
            #将自适应gamma和beta送入到AdaILN
            x = getattr(self, 'UpBlock1_' + str(i+1))(x, gamma, beta)
        out = self.UpBlock2(x) #通过上采样后的模块，得到生成结果

        return out, cam_logit, heatmap #模型输出为生成结果，cam预测以及热力图


class ResnetBlock(nn.Module): #编码器中的残差块
    def __init__(self, dim, use_bias):
        super(ResnetBlock, self).__init__()
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
                       nn.InstanceNorm2d(dim),
                       nn.ReLU(True)]

        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
                       nn.InstanceNorm2d(dim)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetAdaILNBlock(nn.Module): #解码器中的自适应残差块
    def __init__(self, dim, use_bias):
        super(ResnetAdaILNBlock, self).__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm1 = adaILN(dim)
        self.relu1 = nn.ReLU(True)

        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm2 = adaILN(dim)

    def forward(self, x, gamma, beta):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, gamma, beta)
        out = self.relu1(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta)

        return out


class adaILN(nn.Module): #Adaptive Layer-Instance Normalization代码
    def __init__(self, num_features, eps=1e-5):
        super(adaILN, self).__init__()
        self.eps = eps
        #adaILN的参数p，通过这个参数来动态调整LN和IN的占比
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1)) 
        self.rho.data.fill_(0.9)

    def forward(self, input, gamma, beta):
        #先求两种规范化的值
        in_mean, in_var = torch.mean(torch.mean(input, dim=2, keepdim=True), dim=3, keepdim=True), torch.var(torch.var(input, dim=2, keepdim=True), dim=3, keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(torch.mean(torch.mean(input, dim=1, keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True), torch.var(torch.var(torch.var(input, dim=1, keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        #合并两种规范化(IN, LN)
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1-self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln 
        #扩张得到结果
        out = out * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)
		
        return out


class ILN(nn.Module): #没有加入自适应的Layer-Instance Normalization，用于上采样
    def __init__(self, num_features, eps=1e-5):
        super(ILN, self).__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.gamma = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.0)
        self.gamma.data.fill_(1.0)
        self.beta.data.fill_(0.0)

    def forward(self, input):
        in_mean, in_var = torch.mean(torch.mean(input, dim=2, keepdim=True), dim=3, keepdim=True), torch.var(torch.var(input, dim=2, keepdim=True), dim=3, keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(torch.mean(torch.mean(input, dim=1, keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True), torch.var(torch.var(torch.var(input, dim=1, keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1-self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        out = out * self.gamma.expand(input.shape[0], -1, -1, -1) + self.beta.expand(input.shape[0], -1, -1, -1)

        return out
```




生成器的代码如上，归结下来有以下几个点：

- 编码器中没有采用AdaILN以及ILN,而且只采用了IN，原文给出了解释：在分类问题中，LN的性能并不比批规范化好，由于辅助分类器与生成器中的编码器连接，为了提高辅助分类器的精度，我们使用实例规范化(批规范化，小批量大小为1)代替AdaLIN；
- 使用类别激活图(CAM)来得到注意力权重；
- 通过注意力特征图得到解码器中AdaILN的gamma和beta；
- 解码器中残差块使用的AdaILN，而其他块使用的是ILN；
- 使用镜像填充，而不是0填充；
- 所有激活函数使用的是RELU。
### 鉴别器




```
class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=5):
        super(Discriminator, self).__init__()
        model = [nn.ReflectionPad2d(1),   #第一层下采样, 尺寸减半(128)，通道数为64
                 nn.utils.spectral_norm(
                 nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=0, bias=True)),
                 nn.LeakyReLU(0.2, True)]

        for i in range(1, n_layers - 2): #第二，三层下采样，尺寸再缩4倍(32)，通道数为256
            mult = 2 ** (i - 1)
            model += [nn.ReflectionPad2d(1),
                      nn.utils.spectral_norm(
                      nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=2, padding=0, bias=True)),
                      nn.LeakyReLU(0.2, True)]

        mult = 2 ** (n_layers - 2 - 1)
        model += [nn.ReflectionPad2d(1), # 尺寸不变（32），通道数为512
                  nn.utils.spectral_norm(
                  nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=1, padding=0, bias=True)),
                  nn.LeakyReLU(0.2, True)]

        # Class Activation Map， 与生成器得类别激活图类似
        mult = 2 ** (n_layers - 2)
        self.gap_fc = nn.utils.spectral_norm(nn.Linear(ndf * mult, 1, bias=False))
        self.gmp_fc = nn.utils.spectral_norm(nn.Linear(ndf * mult, 1, bias=False))
        self.conv1x1 = nn.Conv2d(ndf * mult * 2, ndf * mult, kernel_size=1, stride=1, bias=True)
        self.leaky_relu = nn.LeakyReLU(0.2, True)

        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.utils.spectral_norm(
            nn.Conv2d(ndf * mult, 1, kernel_size=4, stride=1, padding=0, bias=False))

        self.model = nn.Sequential(*model)

    def forward(self, input):
        x = self.model(input)

        gap = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)
        gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

        cam_logit = torch.cat([gap_logit, gmp_logit], 1)
        x = torch.cat([gap, gmp], 1)
        x = self.leaky_relu(self.conv1x1(x))

        heatmap = torch.sum(x, dim=1, keepdim=True)

        x = self.pad(x)
        out = self.conv(x) #输出大小是32x32，其他与生成器类似

        return out, cam_logit, heatmap
```


