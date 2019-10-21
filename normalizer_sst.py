from torch.utils.data import DataLoader
import networkx as nx
import torch.optim as optim
from gn_models import init_graph_features, FFGN, Normalizer
import torch
from tensorboardX import SummaryWriter
from datetime import datetime
import os
from tqdm import tqdm
from dataset import SSTDataset
from utils import load_graph_features_sst_acrobot as load_graph_features
import sys

if __name__ == "__main__":
    print("loading...")
    dset = SSTDataset(sys.argv[1],
                      dim_control=1, dim_state=4)
    use_cuda = True
    dl = DataLoader(dset, batch_size=200, num_workers=0, drop_last=True)
    G1 = nx.path_graph(2).to_directed()

    node_feat_size = 2
    edge_feat_size = 3
    graph_feat_size = 10
    gn = FFGN(graph_feat_size, node_feat_size, edge_feat_size).cuda()
    optimizer = optim.Adam(gn.parameters(), lr=1e-3)
    savedir = os.path.join('./logs', 'runs',
                           datetime.now().strftime('%B%d_%H:%M:%S'))
    writer = SummaryWriter(savedir)
    step = 0

    in_normalizer = Normalizer()
    out_normalizer = Normalizer()

    for epoch in range(1):
        print("epoch: ", epoch)
        for data in tqdm(dl):
            action, delta_state, last_state = data
            action, delta_state, last_state = action.float(),\
                delta_state.float(),\
                last_state.float()
            if use_cuda:
                action, delta_state, last_state = action.cuda(),\
                    delta_state.cuda(), last_state.cuda()

            G1 = init_graph_features(G1, graph_feat_size, node_feat_size,
                                edge_feat_size, cuda=True, bs=200)
            G1 = load_graph_features(G1, action, last_state, None, noise=0,
                                bs=200, norm=True)
            in_normalizer.input(G1)
            load_graph_features(G1, action, delta_state, None, noise=0,
                                bs=200, norm=False)
            out_normalizer.input(G1)

    torch.save({"in_normalizer": in_normalizer,
                "out_normalizer": out_normalizer},
                sys.argv[2])
