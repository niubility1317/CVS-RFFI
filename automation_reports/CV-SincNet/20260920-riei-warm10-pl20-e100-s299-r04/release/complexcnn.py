# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class ComplexConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ComplexConv, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ## Model components
        self.conv_re = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias)
        self.conv_im = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):  # shpae of x : [batch,channel,axis1]
        x_real = x[:, 0:x.shape[1]//2, :]
        x_img = x[:, x.shape[1] // 2 : x.shape[1], :]
        real = self.conv_re(x_real) - self.conv_im(x_img)
        imaginary = self.conv_re(x_img) + self.conv_im(x_real)
        output = torch.cat((real, imaginary), dim=1)
        return output

class ComplexResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ComplexResidualBlock, self).__init__()

        # 第一个复数卷积层
        self.conv1 = ComplexConv(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels * 2)  # 注意：复数输出通道数翻倍

        # 第二个复数卷积层
        self.conv2 = ComplexConv(out_channels, out_channels, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm1d(out_channels * 2)

        # shortcut connection - 保持复数结构
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # 使用1x1卷积调整维度和步长
            self.shortcut = nn.Sequential(
                ComplexConv(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm1d(out_channels * 2)
            )

    def forward(self, x):
        residual = self.shortcut(x)

        # 主路径
        out = self.conv1(x)
        out = F.relu(self.bn1(out))

        out = self.conv2(out)
        out = self.bn2(out)

        # 残差连接
        out += residual
        out = F.relu(out)

        return out
