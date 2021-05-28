import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
import math
import torch.nn.utils.weight_norm as weightNorm
from collections import OrderedDict

from torch.autograd import Function


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)  ##

class feat_bootleneck(nn.Module):  #
    def __init__(self, feature_dim, bottleneck_dim=64, type="ori"):
        super(feat_bootleneck, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
            x = self.dropout(x)
        return x

class feat_classifier(nn.Module):  ##  256 *  class_num
    def __init__(self, class_num, bottleneck_dim, type="linear"):
        super(feat_classifier, self).__init__()
        if type == "linear":
            self.fc = nn.Linear(bottleneck_dim, class_num)
        else:
            self.fc = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
        self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        return x

class feat_classifier2(nn.Module):  ##  256 *  class_num
    def __init__(self, domain_num, bottleneck_dim, type="linear"):
        super(feat_classifier2, self).__init__()
        if type == "linear":

            self.fc = nn.Linear(bottleneck_dim, domain_num)
        else:
            self.fc = weightNorm(nn.Linear(bottleneck_dim, domain_num), name="weight")
        self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        return x


class feat_domain(nn.Module):  ##  256 *  class_num
    def __init__(self, domain_num, bottleneck_dim=64, type="linear"):
        super(feat_domain, self).__init__()
        if type == "linear":
            self.fc = nn.Linear(bottleneck_dim, class_num)
        else:
            self.fc = weightNorm(nn.Linear(bottleneck_dim, domain_num), name="weight")
        self.fc.apply(init_weights)
        self.logsoftmax = nn.Softmax(dim=1)


    def forward(self, x, alpha = 1.0):

        x = ReverseLayerF.apply(x, alpha)
        x = self.fc(x)
        # x = self.logsoftmax(x)
        return x

class DTNBase(nn.Module):
    def __init__(self):
        super(DTNBase, self).__init__()
        self.conv_params = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(64),
                nn.Dropout2d(0.1),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(128),
                nn.Dropout2d(0.3),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(256),
                nn.Dropout2d(0.5),
                nn.ReLU()
                )   
        self.in_features = 256*4*4

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        return x

class LeNetBase(nn.Module):
    def __init__(self):
        super(LeNetBase, self).__init__()
        self.conv_params = nn.Sequential(
                nn.Conv2d(1, 20, kernel_size=5),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Conv2d(20, 50, kernel_size=5),
                nn.Dropout2d(p=0.5),
                nn.MaxPool2d(2),
                nn.ReLU(),
                )
        self.in_features = 50*4*4

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        return x

class SeedBase(nn.Module):
    def __init__(self):
        super(SeedBase,self).__init__()
        self.conv_params = nn.Sequential(
            nn.Linear(310, 128, bias = True),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # nn.Linear(256, 128),
            # nn.BatchNorm1d(128),
            # nn.ReLU(),
            nn.Linear(128,64, bias= True),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # nn.Dropout(0.5)
        )

        self.in_features = 128

    def forward(self, x):
        x = x.reshape(x.size(0), 310).float()
        x = self.conv_params(x)
        return x

class SeedBase2(nn.Module):
    def __init__(self):
        super(SeedBase2,self).__init__()
        self.conv_params = nn.Sequential(
            nn.Linear(310, 128, bias = True),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # nn.Linear(256, 128),
            # nn.BatchNorm1d(128),
            # nn.ReLU(),
            nn.Linear(128,64, bias= True),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # nn.Dropout(0.5)
        )

        self.in_features = 128

    def forward(self, x):
        x = x.reshape(x.size(0), 310).float()
        x = self.conv_params(x)
        return x

