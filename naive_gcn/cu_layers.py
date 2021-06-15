import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import cugraph as cg

class GraphConvolution_cu(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution_cu, self).__init__()
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

    def forward(self, input, G, using_Batch = True):
        # need to change this part
        # input is X, adj is A, self.weight is w
        support = torch.mm(input, self.weight)
        
        # get 1 hop ego net
        output1 = [] 
        # first use stupid method to write then change to batch ego net
        #batched_ego_graphs
        num_nodes = G.number_of_nodes()
        for node in range(num_nodes):
            #ego_Net = nx.ego_graph(G,node)
            ego_Net = cg.ego_graph(G,node,radius=1)
            target= ego_Net.view_edge_list()['dst'].unique()
            #target = [item[1] for item in list(ego_Net.edges(node))]
            #print ("target", target)
            #print ("node idx", node)
            target_list = []
            for i in range (len(target)):
                target_list += [target[i]]
            #print (target_list)
            output_vect = sum(support[target_list])/len(target_list)
            #print (output_vect)
            output1 += [output_vect]
        
        output2 = torch.stack(output1, dim =0)
        print (output2.cpu().detach().numpy())        
        print("done with for loop")
        if self.bias is not None:
            return output2 + self.bias
        else:
            return output2
        

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
