import torch
from  torch import nn
import torch.nn.functional as F
from complexcnn import ComplexConv, ComplexResidualBlock

#CLASS大小，需改动
# class_num = 10
class_num = 10

class StochasticClassifier(nn.Module):
    def __init__(self, num_features, num_classes, temp=0.05):
        super().__init__()
        #
        self.mu = nn.Parameter(0.01 * torch.randn(num_classes, num_features))
        self.sigma = nn.Parameter(torch.zeros(num_classes, num_features))
        self.temp = temp

    def forward(self, x, stochastic=True):
        mu = self.mu
        sigma = self.sigma

        if stochastic:
            sigma = F.softplus(sigma - 4)  # when sigma=0, softplus(sigma-4)=0.0181
            weight = sigma * torch.randn_like(mu) + mu
        else:
            weight = mu
        # 对权重矩阵每行进行L2归一化
        weight = F.normalize(weight, p=2, dim=1)
        # 对输入特征向量每行进行L2归一化
        x = F.normalize(x, p=2, dim=1)

        score = F.linear(x, weight)
        # temp 是温度参数，它在这个softmax分类器中起着至关重要的作用。
        # 温度的作用是“锐化”或“平滑”softmax输出的概率分布。
        score = score / self.temp

        return score
# IQ向量


class Extractor(nn.Module):

    def __init__(self):
        super(Extractor, self).__init__()
        self.conv1 = ComplexConv(in_channels=1,out_channels=64,kernel_size=3)
        #num_feature指输入的通道数
        self.batchnorm1 = nn.BatchNorm1d(num_features=128)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = ComplexConv(in_channels=64,out_channels=64,kernel_size=3)
        self.batchnorm2 = nn.BatchNorm1d(num_features=128)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)
        self.conv21 = ComplexConv(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm21 = nn.BatchNorm1d(num_features=128)

        self.conv3 = ComplexConv(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm3 = nn.BatchNorm1d(num_features=128)
        self.maxpool3 = nn.MaxPool1d(kernel_size=2)
        self.conv31 = ComplexConv(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm31 = nn.BatchNorm1d(num_features=128)

        self.conv4 = ComplexConv(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm4 = nn.BatchNorm1d(num_features=128)
        self.maxpool4 = nn.MaxPool1d(kernel_size=2)

        self.conv5 = ComplexConv(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm5 = nn.BatchNorm1d(num_features=128)
        # self.maxpool5 = nn.MaxPool1d(kernel_size=1)

        self.conv6 = ComplexConv(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm6 = nn.BatchNorm1d(num_features=128)
        self.maxpool6 = nn.MaxPool1d(kernel_size=2)

        self.conv7 = ComplexConv(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm7 = nn.BatchNorm1d(num_features=128)
        self.maxpool7 = nn.MaxPool1d(kernel_size=2)
        self.conv71 = ComplexConv(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm71 = nn.BatchNorm1d(num_features=128)

        self.conv8 = ComplexConv(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm8 = nn.BatchNorm1d(num_features=128)
        self.maxpool8 = nn.MaxPool1d(kernel_size=2)

        self.conv9 = ComplexConv(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm9 = nn.BatchNorm1d(num_features=128)
        self.maxpool9 = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()

        #特征输出大小，需改动
        self.linear1 = nn.LazyLinear(512)
        self.linear2 = nn.LazyLinear(256)

    def forward(self,x):


        x = self.conv1(x)
        x = F.relu(x)
        x = self.batchnorm1(x)
        x = self.maxpool1(x)
        # print(x.shape)  #torch.Size([32, 128, 1023])

        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchnorm2(x)
        x = self.maxpool2(x)
        # print(x.shape)  #torch.Size([32, 128, 510])
        x = self.conv21(x)
        x = F.relu(x)
        x = self.batchnorm21(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchnorm3(x)
        x = self.maxpool3(x)
        # print(x.shape)  #torch.Size([32, 128, 253])
        x = self.conv31(x)
        x = F.relu(x)
        x = self.batchnorm31(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.batchnorm4(x)
        x = self.maxpool4(x)
        # print(x.shape)  #torch.Size([32, 128, 124])

        x = self.conv5(x)
        x = F.relu(x)
        x = self.batchnorm5(x)
        # x = self.maxpool5(x)
        # print(x.shape)  #torch.Size([32, 128, 122])

        x = self.conv6(x)
        x = F.relu(x)
        x = self.batchnorm6(x)
        x = self.maxpool6(x)
        # print(x.shape)  #torch.Size([32, 128, 60])

        x = self.conv7(x)
        x = F.relu(x)
        x = self.batchnorm7(x)
        x = self.maxpool7(x)
        # print(x.shape)  #torch.Size([32, 128, 29])
        x = self.conv71(x)
        x = F.relu(x)
        x = self.batchnorm71(x)

        x = self.conv8(x)
        x = F.relu(x)
        x = self.batchnorm8(x)
        x = self.maxpool8(x)
        # print(x.shape)  #torch.Size([32, 128, 12])

        x = self.conv9(x)
        x = F.relu(x)
        x = self.batchnorm9(x)
        x = self.maxpool9(x)
        # print(x.shape)  #torch.Size([32, 128, 5])

        x = self.flatten(x)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        features = F.relu(x)


        return features



# 普通分类器

# class Classifier(nn.Module):
#     def __init__(self):
#         super(Classifier, self).__init__()
#         #特征转换情况，需改动
#         #self.linear = nn.Linear(128,class_num)
#         self.linear = nn.Linear(128,64)
#         self.linear2 = nn.Linear(64,32)
#         self.linear3 = nn.Linear(32,class_num)

#     def forward(self,x):
#         x = self.linear(x)
#         x = F.relu(x)
#         # x = self.dro1(x)

#         x = self.linear2(x)
#         x = F.relu(x)
#         # x = self.dro2(x)

#         x = self.linear3(x)
#         return x

# SNN  直接分类
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.linear1 = StochasticClassifier(128,class_num)

    def forward(self,x,stochastic=True):
        x = self.linear1(x,stochastic)
        return x

class Ni_model_2part(nn.Module):
    def __init__(self):
        super(Ni_model_2part, self).__init__()
        self.encoder = Extractor()
        self.classifier = Classifier()

    def forward(self, x,stochastic=True):

        feature = self.encoder(x)
        feature_domain_relevant = feature[:, :128]
        feature_domain_invariant = feature[:, 128:]
        # # SNN
        x = self.classifier(feature_domain_invariant,stochastic)
        # normal
        # x = self.classifier(feature)

        return feature, x


# 普通域分类器
class Domain_Classifier(nn.Module):
    def __init__(self, domain_class=4):
        super(Domain_Classifier, self).__init__()
        #特征转换情况，需改动
        #self.linear = nn.Linear(128,class_num)
        self.linear = nn.Linear(128,64)
        # self.dro1 = nn.Dropout(p=0.1)
        self.linear2 = nn.Linear(64,32)
        # self.dro2 = nn.Dropout(p=0.2)
        self.linear3 = nn.Linear(32,16)
        # self.dro3 = nn.Dropout(p=0.2)
        # 不带SNN
        self.linear4 = nn.Linear(16,domain_class)

    def forward(self,x):
        x = self.linear(x)
        x = F.relu(x)
        # x = self.dro1(x)

        x = self.linear2(x)
        x = F.relu(x)
        # x = self.dro2(x)

        x = self.linear3(x)
        x = F.relu(x)
        # x = self.dro3(x)

        # 不带SNN
        x = self.linear4(x)
        return x

if __name__ == "__main__":
    model = Ni_model_2part()
    input = torch.randn((32,2,2048))

    output= model.encoder(input)
    print(input.shape)
    print(output.shape)
    output = model.classifier(torch.randn((32,128)))
    print(output.shape)
