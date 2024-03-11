from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import math
import gcn3d 

class NetVLADLoupe(nn.Module):
    def __init__(self, feature_size, max_samples, cluster_size, output_dim,
                 gating=True, add_batch_norm=True, is_training=True):
        super(NetVLADLoupe, self).__init__()
        self.feature_size = feature_size
        self.max_samples = max_samples
        self.output_dim = output_dim
        self.is_training = is_training
        self.gating = gating
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size
        self.softmax = nn.Softmax(dim=-1)
        self.cluster_weights = nn.Parameter(torch.randn(
            feature_size, cluster_size) * 1 / math.sqrt(feature_size))
        self.cluster_weights2 = nn.Parameter(torch.randn(
            1, feature_size, cluster_size) * 1 / math.sqrt(feature_size))
        self.hidden1_weights = nn.Parameter(
            torch.randn(cluster_size * feature_size, output_dim) * 1 / math.sqrt(feature_size))

        if add_batch_norm:
            self.cluster_biases = None
            self.bn1 = nn.BatchNorm1d(cluster_size)
        else:
            self.cluster_biases = nn.Parameter(torch.randn(
                cluster_size) * 1 / math.sqrt(feature_size))
            self.bn1 = None

        self.bn2 = nn.BatchNorm1d(output_dim)

        if gating:
            self.context_gating = GatingContext(
                output_dim, add_batch_norm=add_batch_norm)

    def forward(self, x):
        x = x.transpose(1, 3).contiguous()
        x = x.reshape((-1, self.max_samples, self.feature_size))
        activation = torch.matmul(x, self.cluster_weights)
        if self.add_batch_norm:

            activation = activation.reshape(-1, self.cluster_size)
            activation = self.bn1(activation)
            activation = activation.reshape(-1,
                                         self.max_samples, self.cluster_size)

        else:
            activation = activation + self.cluster_biases
        activation = self.softmax(activation)
        activation = activation.reshape((-1, self.max_samples, self.cluster_size))

        a_sum = activation.sum(-2, keepdim=True)
        a = a_sum * self.cluster_weights2

        activation = torch.transpose(activation, 2, 1)
        x = x.reshape((-1, self.max_samples, self.feature_size))
        vlad = torch.matmul(activation, x)
        vlad = torch.transpose(vlad, 2, 1)
        vlad = vlad - a

        vlad = F.normalize(vlad, dim=1, p=2)
        vlad = vlad.reshape((-1, self.cluster_size * self.feature_size))
        vlad = F.normalize(vlad, dim=1, p=2)

        vlad = torch.matmul(vlad, self.hidden1_weights)

        vlad = self.bn2(vlad)

        if self.gating:
            vlad = self.context_gating(vlad)

        return vlad


class GatingContext(nn.Module):
    def __init__(self, dim, add_batch_norm=True):
        super(GatingContext, self).__init__()
        self.dim = dim
        self.add_batch_norm = add_batch_norm
        self.gating_weights = nn.Parameter(
            torch.randn(dim, dim) * 1 / math.sqrt(dim))
        self.sigmoid = nn.Sigmoid()

        if add_batch_norm:
            self.gating_biases = None
            self.bn1 = nn.BatchNorm1d(dim)
        else:
            self.gating_biases = nn.Parameter(
                torch.randn(dim) * 1 / math.sqrt(dim))
            self.bn1 = None

    def forward(self, x):
        gates = torch.matmul(x, self.gating_weights)

        if self.add_batch_norm:
            gates = self.bn1(gates)
        else:
            gates = gates + self.gating_biases

        gates = self.sigmoid(gates)

        activation = x * gates

        return activation


class Flatten(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, input):
        return input.reshape(input.size(0), -1)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):

        x = x.clamp(min=self.eps).pow(self.p)


        x = x.mean(dim=-2, keepdim=False)

        return x.pow(1. / self.p).squeeze(-1)



class StructuralBlock(nn.Module):
    def __init__(self, support_num: int, neighbor_num: int):
        super().__init__()
        self.neighbor_num = neighbor_num

        self.conv_0 = gcn3d.Conv_surface(kernel_num= 32, support_num= support_num)
        self.conv_1 = gcn3d.Conv_layer(32, 64, support_num= support_num)
        self.pool_1 = gcn3d.Pool_layer(pooling_rate= 4, neighbor_num= 4)
        self.conv_2 = gcn3d.Conv_layer(64, 128, support_num= support_num)
        self.conv_3 = gcn3d.Conv_layer(128, 256, support_num= support_num)
        self.pool_2 = gcn3d.Pool_layer(pooling_rate= 4, neighbor_num= 4)
        self.conv_4 = gcn3d.Conv_layer(256, 1024, support_num= support_num)


    def forward(self,  vertices: "(bs, 1, vertice_num, 3)"):
        bs, _, vertice_num, _ = vertices.size()
        vertices = vertices.view(bs, vertice_num, 3) 
 

        neighbor_index = gcn3d.get_neighbor_index(vertices, self.neighbor_num)


        fm_0 = self.conv_0(neighbor_index, vertices)
        fm_0 = F.relu(fm_0, inplace= True)
        fm_1 = self.conv_1(neighbor_index, vertices, fm_0)
        fm_1 = F.relu(fm_1, inplace= True)
        vertices, fm_1 = self.pool_1(vertices, fm_1,neighbor_index )
        neighbor_index = gcn3d.get_neighbor_index(vertices, self.neighbor_num)

        fm_2 = self.conv_2(neighbor_index, vertices, fm_1)
        fm_2 = F.relu(fm_2, inplace= True) 
        fm_3 = self.conv_3(neighbor_index, vertices, fm_2)
        fm_3 = F.relu(fm_3, inplace= True) 
        vertices, fm_3 = self.pool_2(vertices, fm_3,neighbor_index  )
        neighbor_index = gcn3d.get_neighbor_index(vertices, self.neighbor_num)
        
        fm_4 = self.conv_4(neighbor_index, vertices, fm_3)

        fm_4 = fm_4.permute(0, 2, 1)  
        fm_4 = fm_4.unsqueeze(-1) 
        return fm_4


class SBDD(nn.Module):
    def __init__(self, num_points=900,  output_dim=1024):
        super(SBDD, self).__init__()
        self.sbdd =StructuralBlock(support_num= 1, neighbor_num= 20)
        self.net_vlad = NetVLADLoupe(feature_size=1024, max_samples=256, cluster_size=64,
                                     output_dim=output_dim, gating=True, add_batch_norm=True,
                                     is_training=True)
        self.gem = GeM()
    def forward(self, x):
        x = self.sbdd(x)
        y=self.gem(x)
        x = self.net_vlad(x)

        return  y , x
