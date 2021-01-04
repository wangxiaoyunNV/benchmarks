import dgl
import torch as th
from dgl.data import RedditDataset
from dgl.data.rdf import AIFBDataset, MUTAGDataset, BGSDataset, AMDataset
import scipy.io
import urllib.request


def load_reddit():
    # load reddit data
    data = RedditDataset(self_loop=True)
    g = data[0]
    g.ndata['features'] = g.ndata['feat']
    g.ndata['labels'] = g.ndata['label']
    return g, data.num_labels


def load_acm():
    # load acm data
    data_url = 'https://data.dgl.ai/dataset/ACM.mat'
    data_file_path = '/tmp/ACM.mat'
    urllib.request.urlretrieve(data_url, data_file_path)
    data = scipy.io.loadmat(data_file_path)
    return data


def load_aifb():
    # load aifb dataset
    dataset = AIFBDataset()
    return dataset

def load_MUTAG():
    # load MUTAGDataset
    dataset = MUTAGDataset()
    return dataset

def load_BGS():
    # load BGSDataset 
    dataset = BGSDataset()
    return dataset

def load_AM():
    # load AMDataset 
    dataset = AMDataset()
    return dataset




if __name__ == '__main__':

    reddit_g, num_labels = load_reddit()
    acm_data = load_acm()
    aifb_data = load_aifb()
    mutagdata = load_MUTAG()
    dgsdata = load_BGS()
    #amdata = load_AM()
    # AM dataset is too slow. commented out first





