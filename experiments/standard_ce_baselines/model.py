"""Common full-width architectures adapted to 1-D IQ, without pretrained weights.

CNN block schedules and widths follow torchvision; 2-D kernels/pools become 1-D.
Sequence models use nonoverlapping four-sample tokens, preserving all 256 samples.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F

COMMON = ('resnet18', 'resnet34', 'resnet50', 'vgg16_bn', 'densenet121',
          'convnext_tiny', 'transformer6', 'bilstm3', 'cvresnet18_1d', 'cvresnet18_2d',
          'resnet18_2d', 'resnet50_2d')
VARIANTS = (*COMMON, 'native', 'residual_fusion')


def conv(a, b, k=3, stride=1, groups=1):
    return nn.Conv1d(a, b, k, stride, k//2, groups=groups, bias=False)


class ResidualBlock(nn.Module):
    def __init__(self, incoming, width, stride, bottleneck):
        super().__init__()
        outgoing = width * (4 if bottleneck else 1)
        if bottleneck:
            self.path = nn.Sequential(conv(incoming, width, 1), nn.BatchNorm1d(width), nn.ReLU(),
                conv(width, width, 3, stride), nn.BatchNorm1d(width), nn.ReLU(),
                conv(width, outgoing, 1), nn.BatchNorm1d(outgoing))
        else:
            self.path = nn.Sequential(conv(incoming, width, 3, stride), nn.BatchNorm1d(width),
                nn.ReLU(), conv(width, width), nn.BatchNorm1d(width))
        self.skip = (nn.Identity() if stride == 1 and incoming == outgoing else
                     nn.Sequential(conv(incoming, outgoing, 1, stride), nn.BatchNorm1d(outgoing)))

    def forward(self, x):
        return F.relu(self.path(x) + self.skip(x))


class ResNet(nn.Module):
    def __init__(self, schedule, bottleneck=False):
        super().__init__()
        blocks = [conv(2, 64, 7, 2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(3, 2, 1)]
        incoming = 64
        for stage, (width, count) in enumerate(zip((64,128,256,512), schedule)):
            for index in range(count):
                blocks.append(ResidualBlock(incoming, width, 2 if stage and index == 0 else 1, bottleneck))
                incoming = width * (4 if bottleneck else 1)
        self.features = nn.Sequential(*blocks)
        self.classifier = nn.Linear(incoming, 6)
        for layer in self.modules():
            if isinstance(layer, nn.Conv1d): nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        features=self.features(x)
        return self.classifier(features.flatten(2).mean(-1))


class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        layers=[]; incoming=2
        for value in (64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M'):
            if value == 'M': layers.append(nn.MaxPool1d(2,2))
            else:
                layers.extend((nn.Conv1d(incoming,value,3,padding=1),nn.BatchNorm1d(value),nn.ReLU()))
                incoming=value
        self.features=nn.Sequential(*layers)
        self.pool=nn.AdaptiveAvgPool1d(7)
        self.classifier=nn.Sequential(nn.Linear(512*7,4096),nn.ReLU(),nn.Dropout(.5),
                                     nn.Linear(4096,4096),nn.ReLU(),nn.Dropout(.5),nn.Linear(4096,6))
        for layer in self.modules():
            if isinstance(layer,nn.Conv1d):
                nn.init.kaiming_normal_(layer.weight,mode='fan_out',nonlinearity='relu');nn.init.zeros_(layer.bias)
            elif isinstance(layer,nn.Linear): nn.init.normal_(layer.weight,0,.01);nn.init.zeros_(layer.bias)

    def forward(self,x): return self.classifier(self.pool(self.features(x)).flatten(1))


class DenseLayer(nn.Module):
    def __init__(self,incoming):
        super().__init__()
        self.path=nn.Sequential(nn.BatchNorm1d(incoming),nn.ReLU(),conv(incoming,128,1),
                                nn.BatchNorm1d(128),nn.ReLU(),conv(128,32))

    def forward(self,x): return torch.cat((x,self.path(x)),1)


class DenseNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers=[conv(2,64,7,2),nn.BatchNorm1d(64),nn.ReLU(),nn.MaxPool1d(3,2,1)]
        incoming=64
        for stage,count in enumerate((6,12,24,16)):
            for _ in range(count): layers.append(DenseLayer(incoming));incoming+=32
            if stage<3:
                layers.append(nn.Sequential(nn.BatchNorm1d(incoming),nn.ReLU(),conv(incoming,incoming//2,1),nn.AvgPool1d(2,2)))
                incoming//=2
        layers.extend((nn.BatchNorm1d(incoming),nn.ReLU()))
        self.features=nn.Sequential(*layers);self.classifier=nn.Linear(incoming,6)
        for layer in self.modules():
            if isinstance(layer,nn.Conv1d):nn.init.kaiming_normal_(layer.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self,x):return self.classifier(self.features(x).mean(-1))


class ChannelNorm(nn.LayerNorm):
    def forward(self,x):return super().forward(x.transpose(1,2)).transpose(1,2)


class ConvNeXtBlock(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.depthwise=nn.Conv1d(width,width,7,padding=3,groups=width)
        self.norm=nn.LayerNorm(width,eps=1e-6)
        self.mlp=nn.Sequential(nn.Linear(width,4*width),nn.GELU(),nn.Linear(4*width,width))
        self.gamma=nn.Parameter(torch.full((width,),1e-6))

    def forward(self,x):
        y=self.depthwise(x).transpose(1,2)
        return x+(self.mlp(self.norm(y))*self.gamma).transpose(1,2)


class ConvNeXt(nn.Module):
    def __init__(self):
        super().__init__()
        layers=[nn.Conv1d(2,96,4,stride=4),ChannelNorm(96,eps=1e-6)]
        for stage,(width,count) in enumerate(zip((96,192,384,768),(3,3,9,3))):
            layers.extend(ConvNeXtBlock(width) for _ in range(count))
            if stage<3:layers.extend((ChannelNorm(width,eps=1e-6),nn.Conv1d(width,2*width,2,stride=2)))
        self.features=nn.Sequential(*layers)
        self.classifier=nn.Sequential(nn.LayerNorm(768,eps=1e-6),nn.Linear(768,6))
        for layer in self.modules():
            if isinstance(layer,(nn.Conv1d,nn.Linear)):
                nn.init.trunc_normal_(layer.weight,std=.02)
                if layer.bias is not None:nn.init.zeros_(layer.bias)

    def forward(self,x):return self.classifier(self.features(x).mean(-1))


def tokens(x):
    return x.transpose(1,2).reshape(x.shape[0],64,8)


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding=nn.Linear(8,512)
        position=torch.arange(64).unsqueeze(1)
        frequency=torch.exp(torch.arange(0,512,2)*(-math.log(10000.)/512))
        encoding=torch.zeros(64,512);encoding[:,0::2]=torch.sin(position*frequency);encoding[:,1::2]=torch.cos(position*frequency)
        self.register_buffer('position',encoding.unsqueeze(0))
        layer=nn.TransformerEncoderLayer(512,8,2048,dropout=.1,batch_first=True,activation='relu',norm_first=False)
        self.encoder=nn.TransformerEncoder(layer,6,enable_nested_tensor=False)
        # Avoid identical initial parameters in cloned TransformerEncoder layers.
        for block in self.encoder.layers:
            nn.init.xavier_uniform_(block.self_attn.in_proj_weight)
            for linear in (block.self_attn.out_proj,block.linear1,block.linear2):
                nn.init.xavier_uniform_(linear.weight);nn.init.zeros_(linear.bias)
        self.classifier=nn.Linear(512,6)

    def forward(self,x):return self.classifier(self.encoder(self.embedding(tokens(x))+self.position).mean(1))


class BiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.LSTM(8,512,3,batch_first=True,bidirectional=True,dropout=.1)
        self.classifier=nn.Linear(1024,6)

    def forward(self,x):
        _,(hidden,_)=self.encoder(tokens(x))
        return self.classifier(torch.cat((hidden[-2],hidden[-1]),1))


class IQGrid(nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,x):return self.model(x.reshape(x.shape[0],2,16,16))


def to_2d(module):
    for name,child in list(module.named_children()):
        if isinstance(child,nn.Conv1d):
            replacement=nn.Conv2d(child.in_channels,child.out_channels,child.kernel_size[0],
                child.stride[0],child.padding[0],groups=child.groups,bias=child.bias is not None)
            nn.init.kaiming_normal_(replacement.weight,mode='fan_out',nonlinearity='relu')
        elif isinstance(child,nn.BatchNorm1d):replacement=nn.BatchNorm2d(child.num_features)
        elif isinstance(child,nn.MaxPool1d):replacement=nn.MaxPool2d(child.kernel_size,child.stride,child.padding)
        else:to_2d(child);continue
        setattr(module,name,replacement)
    return module


class ComplexConv(nn.Module):
    def __init__(self,incoming,outgoing,kernel,dimensions,stride=1):
        super().__init__();self.width=outgoing
        cls=nn.Conv1d if dimensions==1 else nn.Conv2d
        self.real=cls(incoming,outgoing,kernel,stride=stride,padding=kernel//2,bias=False)
        self.imag=cls(incoming,outgoing,kernel,stride=stride,padding=kernel//2,bias=False)
        for weight in (self.real.weight,self.imag.weight):
            nn.init.kaiming_normal_(weight,mode='fan_in',nonlinearity='relu')
            with torch.no_grad():weight.div_(math.sqrt(2))
    def forward(self,x):
        real,imag=x.chunk(2,1)
        return torch.cat((self.real(real)-self.imag(imag),self.real(imag)+self.imag(real)),1)


def to_complex(module,dimensions):
    """Published common approach: convert a standard ResNet to complex layers.

    Preserve [2,2,2,2] blocks and [64,128,256,512] complex widths. Use the
    split-real/imag BN/CReLU option documented in complexPyTorch; final head
    classifies both real/imag features. This is an IQ adaptation, not an exact
    reproduction of torchcvnn's covariance-BN/activation configuration.
    """
    for name,child in list(module.named_children()):
        if isinstance(child,nn.Conv1d):
            replacement=ComplexConv(1 if child.in_channels==2 else child.in_channels,
                child.out_channels,child.kernel_size[0],dimensions,child.stride[0])
        elif isinstance(child,nn.BatchNorm1d):
            cls=nn.BatchNorm1d if dimensions==1 else nn.BatchNorm2d
            replacement=cls(2*child.num_features)
        elif isinstance(child,nn.MaxPool1d):
            cls=nn.MaxPool1d if dimensions==1 else nn.MaxPool2d
            replacement=cls(child.kernel_size,child.stride,child.padding)
        elif isinstance(child,nn.Linear):replacement=nn.Linear(2*child.in_features,child.out_features)
        else:to_complex(child,dimensions);continue
        setattr(module,name,replacement)
    return module


def build(variant):
    if variant=='native':
        from experiments.cvs_identity_ce.model import IdentityOnlyCVS
        return IdentityOnlyCVS()
    if variant=='residual_fusion':
        from experiments.cvs_residual_identity.model import build as cvs
        return cvs(variant)
    factories={'resnet18':lambda:ResNet((2,2,2,2)), 'resnet34':lambda:ResNet((3,4,6,3)),
               'resnet50':lambda:ResNet((3,4,6,3),True),'vgg16_bn':VGG,'densenet121':DenseNet,
               'convnext_tiny':ConvNeXt,'transformer6':Transformer,'bilstm3':BiLSTM}
    factories.update(cvresnet18_1d=lambda:to_complex(ResNet((2,2,2,2)),1),
        cvresnet18_2d=lambda:IQGrid(to_complex(ResNet((2,2,2,2)),2)),
        resnet18_2d=lambda:IQGrid(to_2d(ResNet((2,2,2,2)))),
        resnet50_2d=lambda:IQGrid(to_2d(ResNet((3,4,6,3),True))))
    if variant not in factories:raise ValueError('Unregistered architecture')
    return factories[variant]()
