import torch.nn as nn
import torch.nn.functional as F
from cu_layers import GraphConvolution_cu


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution_cu(nfeat, nhid)
        self.gc2 = GraphConvolution_cu(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        #print ("x,", sum(x).cpu().numpy())
        x = F.relu(self.gc1(x, adj))
        print("done gc1",x.cpu().detach().numpy())
        x = F.dropout(x, self.dropout, training=self.training)
        print("done dropout, x, adj",x,adj)
        x = self.gc2(x, adj)
        print("done gc2", x.cpu().detach().numpy())
        return F.log_softmax(x, dim=1)
