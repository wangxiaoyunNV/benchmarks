import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import networkx as nx


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, G):
        # need to change this part
        # input is X, adj is A, self.weight is w
        support = torch.mm(input, self.weight)
        #adj_mat = adj.to_dense().cpu().numpy()
        #G = nx.from_numpy_matrix(adj_mat)
        # get 1 hop ego net
        output1 = [] 
        for node in G.nodes():
            ego_Net = nx.ego_graph(G,node)
            target = [item[1] for item in list(ego_Net.edges(node))]
            output1 += [output_vect]
            output2 = torch.stack(output1, dim =0)

        
        if self.bias is not None:
            return output2 + self.bias
        else:
            return output2

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
