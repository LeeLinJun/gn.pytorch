from torch.utils.data import DataLoader
import networkx as nx
import torch.optim as optim
# import matplotlib.pyplot as plt
from gn_models import init_graph_features, FFGN
import torch
from tensorboardX import SummaryWriter
from datetime import datetime
import os
from dataset import SSTDataset
from utils import load_graph_features_sst_acrobot as load_graph_features,\
    get_graph_loss
from tqdm import tqdm
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='',  help='model path')
    opt = parser.parse_args()
    print(opt)

    dset = SSTDataset('data/acrobot.npy',
                      dim_control=1, dim_state=4)
    dset_eval = SSTDataset('data/acrobot_test.npy',
                           dim_control=1, dim_state=4)
    use_cuda = True

    dl = DataLoader(dset, batch_size=200, num_workers=0, drop_last=True)
    dl_eval = DataLoader(dset_eval, batch_size=200, num_workers=0,
                         drop_last=True)

    G1 = nx.path_graph(2).to_directed()
    G_target = nx.path_graph(2).to_directed()
    # nx.draw(G1)
    # plt.show()
    node_feat_size = 2
    edge_feat_size = 3
    graph_feat_size = 10
    gn = FFGN(graph_feat_size, node_feat_size, edge_feat_size).cuda()
    if opt.model != '':
        gn.load_state_dict(torch.load(opt.model))

    optimizer = optim.Adam(gn.parameters(), lr=1e-4)
    schedular = optim.lr_scheduler.StepLR(optimizer, 5e4, gamma=0.975)
    savedir = os.path.join('./logs', 'runs',
                           datetime.now().strftime('%B%d_%H:%M:%S'))
    writer = SummaryWriter(savedir)
    step = 0

    normalizers = torch.load('normalized/acrobot.pth')
    in_normalizer = normalizers['in_normalizer']
    out_normalizer = normalizers['out_normalizer']
    std = in_normalizer.get_std()
    for epoch in range(300):
        with tqdm(dl, total=len(dset) / 200 + 1) as pbar:
            for data in pbar:
                optimizer.zero_grad()
                action, delta_state, last_state = data
                action, delta_state, last_state = action.float(),\
                    delta_state.float(), last_state.float()
                if use_cuda:
                    action, delta_state, last_state = action.cuda(),\
                        delta_state.cuda(), last_state.cuda()

                init_graph_features(G1, graph_feat_size, node_feat_size,
                                    edge_feat_size, cuda=True, bs=200)
                load_graph_features(G1, action, last_state, delta_state, bs=200,
                                    noise=0.00, std=std)
                G_out = gn(in_normalizer.normalize(G1))

                init_graph_features(G_target, graph_feat_size, node_feat_size,
                                    edge_feat_size, cuda=True, bs=200)
                load_graph_features(G_target, action, delta_state, None, bs=200,
                                    norm=False, noise=0)
                #G_target_normalized = out_normalizer.normalize(G_target)

                loss = get_graph_loss(G_out, G_target)#G_target_normalized)
                loss.backward()
                if step % 10 == 1:
                    writer.add_scalar('loss', loss.data.item(), step)
                    pbar.set_postfix({'loss' : '{0:1.5f}'.format(loss.data.item())})
                step += 1
                for param in gn.parameters():
                    if param.grad is not None:
                        param.grad.clamp_(-3, 3)

                optimizer.step()
                schedular.step()
                if step % 10 == 0:
                    torch.save(
                        gn.state_dict(),
                        savedir +
                        '/model_{}.pth'.format(step))

                pbar.update(1)

        itr = 0
        sum_loss = 0

        # evaluation loop, done every epoch

        for data in tqdm(dl_eval):
            action, delta_state, last_state = data
            action, delta_state, last_state = action.float(), \
                delta_state.float(), last_state.float()
            if use_cuda:
                action, delta_state, last_state = action.cuda(),\
                    delta_state.cuda(), last_state.cuda()

            init_graph_features(G1, graph_feat_size, node_feat_size,
                                edge_feat_size, cuda=True, bs=200)
            load_graph_features(G1, action, last_state, delta_state,
                                bs=200, noise=0)
            G_out = gn(in_normalizer.normalize(G1))

            init_graph_features(G_target, graph_feat_size, node_feat_size,
                                edge_feat_size, cuda=True, bs=200)
            load_graph_features(G_target, action, delta_state, None, bs=200,
                                norm=False, noise=0)
            #G_target_normalized = out_normalizer.normalize(G_target)

            loss = get_graph_loss(G_out, G_target)#G_target_normalized)
            sum_loss += loss.data.item()
            itr += 1

        writer.add_scalar('loss_eval', sum_loss / float(itr), step)
