# Modified 3dgcn code: https://https://github.com/zhihao-lin/3dgcn
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_neighbor_index(vertices: "(bs, vertice_num, 3)",  neighbor_num: int):
    """
    Return: (bs, vertice_num, neighbor_num)
    """
    bs, v, _ = vertices.size()
    device = vertices.device
    inner = torch.bmm(vertices, vertices.transpose(1, 2)) #(bs, v, v)
    quadratic = torch.sum(vertices**2, dim= 2) #(bs, v)
    distance = inner * (-2) + quadratic.unsqueeze(1) + quadratic.unsqueeze(2)
    neighbor_index = torch.topk(distance, k= neighbor_num + 1, dim= -1, largest= False)[1]
    neighbor_index = neighbor_index[:, :, 1:]
    return neighbor_index

def get_nearest_index(target: "(bs, v1, 3)", source: "(bs, v2, 3)"):
    """
    Return: (bs, v1, 1)
    """
    inner = torch.bmm(target, source.transpose(1, 2)) #(bs, v1, v2)
    s_norm_2 = torch.sum(source ** 2, dim= 2) #(bs, v2)
    t_norm_2 = torch.sum(target ** 2, dim= 2) #(bs, v1)
    d_norm_2 = s_norm_2.unsqueeze(1) + t_norm_2.unsqueeze(2) - 2 * inner
    nearest_index = torch.topk(d_norm_2, k= 1, dim= -1, largest= False)[1]
    return nearest_index

def indexing_neighbor(tensor: "(bs, vertice_num, dim)", index: "(bs, vertice_num, neighbor_num)" ):
    """
    Return: (bs, vertice_num, neighbor_num, dim)
    """
    bs, v, n = index.size()
    id_0 = torch.arange(bs).view(-1, 1, 1)
    tensor_indexed = tensor[id_0, index]
    return tensor_indexed

def get_neighbor_direction_norm(vertices: "(bs, vertice_num, 3)", neighbor_index: "(bs, vertice_num, neighbor_num)"):
    """
    Return: (bs, vertice_num, neighobr_num, 3)
    """
    neighbors = indexing_neighbor(vertices, neighbor_index) # (bs, v, n, 3)
    neighbor_direction = neighbors - vertices.unsqueeze(2)
    neighbor_direction_norm = F.normalize(neighbor_direction, dim= -1)
    return neighbor_direction_norm

class Conv_surface(nn.Module):
    """Extract structure feafure from surface, independent from vertice coordinates"""
    def __init__(self, kernel_num, support_num):
        super().__init__()
        self.kernel_num = kernel_num
        self.support_num = support_num

        self.relu = nn.ReLU(inplace= True)
        self.directions = nn.Parameter(torch.FloatTensor(3, support_num * kernel_num))
        self.initialize()

    def initialize(self):
        stdv = 1. / math.sqrt(self.support_num * self.kernel_num)
        self.directions.data.uniform_(-stdv, stdv)
    
    def forward(self, 
                neighbor_index: "(bs, vertice_num, neighbor_num)", 
                vertices: "(bs, vertice_num, 3)"):
        """
        Return vertices with local feature: (bs, vertice_num, kernel_num)
        """
        bs, vertice_num, neighbor_num = neighbor_index.size() 
        neighbor_direction_norm = get_neighbor_direction_norm(vertices, neighbor_index)
        support_direction_norm = F.normalize(self.directions, dim= 0) #(3, s * k)
        theta = neighbor_direction_norm @ support_direction_norm # (bs, vertice_num, neighbor_num, s*k)

        theta = self.relu(theta)
        theta = theta.contiguous().view(bs, vertice_num, neighbor_num, self.support_num, self.kernel_num)
        theta = torch.max(theta, dim= 2)[0] # (bs, vertice_num, support_num, kernel_num)
        feature = torch.sum(theta, dim= 2) # (bs, vertice_num, kernel_num)
        return feature

class Conv_layer(nn.Module):
    def __init__(self, in_channel, out_channel, support_num):
        super().__init__()
        # arguments: 
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.support_num = support_num

        # parameters:
        self.relu = nn.ReLU(inplace= True)
        self.weights = nn.Parameter(torch.FloatTensor(in_channel, (support_num + 1) * out_channel))
        self.bias = nn.Parameter(torch.FloatTensor((support_num + 1) * out_channel))
        self.directions = nn.Parameter(torch.FloatTensor(3, support_num * out_channel))
        self.initialize()

    def initialize(self):
        stdv = 1. / math.sqrt(self.out_channel * (self.support_num + 1))
        self.weights.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)
        self.directions.data.uniform_(-stdv, stdv)

    def forward(self, 
                neighbor_index: "(bs, vertice_num, neighbor_index)",
                vertices: "(bs, vertice_num, 3)",
                feature_map: "(bs, vertice_num, in_channel)"):
        """
        Return: output feature map: (bs, vertice_num, out_channel)
        """
        bs, vertice_num, neighbor_num = neighbor_index.size()
        neighbor_direction_norm = get_neighbor_direction_norm(vertices, neighbor_index)
        support_direction_norm = F.normalize(self.directions, dim= 0)
        theta = neighbor_direction_norm @ support_direction_norm # (bs, vertice_num, neighbor_num, support_num * out_channel)
        theta = self.relu(theta)
        theta = theta.contiguous().view(bs, vertice_num, neighbor_num, -1)
        # (bs, vertice_num, neighbor_num, support_num * out_channel)

        feature_out = feature_map @ self.weights + self.bias # (bs, vertice_num, (support_num + 1) * out_channel)
        feature_center = feature_out[:, :, :self.out_channel] # (bs, vertice_num, out_channel)
        feature_support = feature_out[:, :, self.out_channel:] #(bs, vertice_num, support_num * out_channel)

        # Fuse together - max among product
        feature_support = indexing_neighbor(feature_support, neighbor_index) # (bs, vertice_num, neighbor_num, support_num * out_channel)
        activation_support = theta * feature_support # (bs, vertice_num, neighbor_num, support_num * out_channel)
        activation_support = activation_support.view(bs,vertice_num, neighbor_num, self.support_num, self.out_channel)
        activation_support = torch.max(activation_support, dim= 2)[0] # (bs, vertice_num, support_num, out_channel)
        activation_support = torch.sum(activation_support, dim= 2)    # (bs, vertice_num, out_channel)
        feature_fuse = feature_center + activation_support # (bs, vertice_num, out_channel)
        return feature_fuse

class Pool_layer(nn.Module):
    def __init__(self, pooling_rate: int= 4, neighbor_num: int=  4):
        super().__init__()
        self.pooling_rate = pooling_rate
        self.neighbor_num = neighbor_num
        self.num_bins = 3 

    def forward(self,
                vertices: "(bs, vertice_num, 3)",
                feature_map: "(bs, vertice_num, channel_num)",
                neighbor_index_density):
        """
        Return:
            vertices_pool: (bs, pool_vertice_num, 3),
            feature_map_pool: (bs, pool_vertice_num, channel_num)
        """
        bs, vertice_num, _ = vertices.size()
        neighbor_index = get_neighbor_index(vertices, self.neighbor_num)
        neighbor_feature = indexing_neighbor(feature_map, neighbor_index) #(bs, vertice_num, neighbor_num, channel_num)
        pooled_feature = torch.max(neighbor_feature, dim= 2)[0] #(bs, vertice_num, channel_num)
        
        neighbor_to_vertices = indexing_neighbor(vertices, neighbor_index_density)
        expanded_vertices = vertices.unsqueeze(2).expand(-1, -1, 20, -1)
        squared_diff = (expanded_vertices - neighbor_to_vertices) ** 2
        distances = torch.sqrt(torch.sum(squared_diff, dim=3))
        

        summed_distances = torch.sum(distances, dim=2)


        normalized_distances = self.normalize_distances_within_batch(summed_distances)
       
       
        selected_indices = self.probabilistic_sampling(normalized_distances, vertice_num)  

        vertices_pool = vertices[torch.arange(bs).unsqueeze(-1), selected_indices]
        feature_map_pool = pooled_feature[torch.arange(bs).unsqueeze(-1), selected_indices]

        return vertices_pool, feature_map_pool
    def normalize_distances_within_batch(self,summed_distances):
            min_values, _ = torch.min(summed_distances, dim=1, keepdim=True)
            max_values, _ = torch.max(summed_distances, dim=1, keepdim=True)
            normalized_distances = (summed_distances - min_values) / (max_values - min_values)
            return normalized_distances

    def probabilistic_sampling(self, normalized_distances, vertice_num):
        bs = normalized_distances.size(0)
        total_samples = int(vertice_num / self.pooling_rate)

        bin_edges = torch.linspace(0, 1, self.num_bins + 1, device=normalized_distances.device)
        bin_indices = torch.bucketize(normalized_distances, bin_edges)


        bin_counts = torch.zeros(bs, self.num_bins, device=normalized_distances.device)
        for bin_index in range(1, self.num_bins + 1):
            bin_counts[:, bin_index - 1] = (bin_indices == bin_index).sum(dim=1).float()


        bin_samples = (bin_counts / bin_counts.sum(dim=1, keepdim=True)) * total_samples

        selected_indices = torch.zeros(bs, total_samples, dtype=torch.long, device=normalized_distances.device)
        for i in range(bs):
            sample_count = 0
            for bin_index in range(self.num_bins):
                samples_in_bin = int(bin_samples[i, bin_index])
                if samples_in_bin > 0:

                    bin_mask = (bin_indices[i] == bin_index + 1)
                    chosen_indices = torch.multinomial(bin_mask.float(), samples_in_bin, replacement=False)
                    selected_indices[i, sample_count:sample_count + samples_in_bin] = chosen_indices
                    sample_count += samples_in_bin

        return selected_indices


def test():
    import time
    bs = 8
    v = 1024
    dim = 3
    n = 20
    vertices = torch.randn(bs, v, dim)
    neighbor_index = get_neighbor_index(vertices, n)

    s = 3
    conv_1 = Conv_surface(kernel_num= 32, support_num= s)
    conv_2 = Conv_layer(in_channel= 32, out_channel= 64, support_num= s)
    pool = Pool_layer(pooling_rate= 4, neighbor_num= 4)
    
    print("Input size: {}".format(vertices.size()))
    start = time.time()
    f1 = conv_1(neighbor_index, vertices)
    print("\n[1] Time: {}".format(time.time() - start))
    print("[1] Out shape: {}".format(f1.size()))
    start = time.time()
    f2 = conv_2(neighbor_index, vertices, f1)
    print("\n[2] Time: {}".format(time.time() - start))
    print("[2] Out shape: {}".format(f2.size()))
    start = time.time()
    v_pool, f_pool = pool(vertices, f2)
    print("\n[3] Time: {}".format(time.time() - start))
    print("[3] v shape: {}, f shape: {}".format(v_pool.size(), f_pool.size()))


if __name__ == "__main__":
    test()

