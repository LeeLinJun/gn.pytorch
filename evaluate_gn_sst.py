from gn_models import init_graph_features, FFGN
from utils import load_graph_features_sst_acrobot as load_graph_features
import networkx as nx
from sparse_rrt.systems import Acrobot
import numpy as np
import torch
from matplotlib import pyplot as plt

G1 = nx.path_graph(2).to_directed()

node_feat_size = 2
edge_feat_size = 3
graph_feat_size = 10
gn = FFGN(graph_feat_size, node_feat_size, edge_feat_size).cuda()
gn.load_state_dict(torch.load('model0_1_.pth'))
normalizers = torch.load('normalized/acrobot.pth')
in_normalizer = normalizers['in_normalizer']
out_normalizer = normalizers['out_normalizer']

model = Acrobot()


action = [0.5]

start_state = np.array([np.random.uniform(low=model.MIN_ANGLE,
                                          high=model.MAX_ANGLE),
                       np.random.uniform(low=model.MIN_ANGLE,
                                         high=model.MAX_ANGLE),
                       np.random.uniform(low=model.MIN_V_1,
                                         high=model.MAX_V_1),
                       np.random.uniform(low=model.MIN_V_2,
                                         high=model.MAX_V_2)])
maxframe = 20
baseline = np.zeros((maxframe+1, 4))
baseline[0, :] = start_state
last_state = start_state.copy()
for i in range(1, maxframe+1):
    new_state = model.propagate(last_state, action,
                                1,  # np.random.randint(low=20, high=200),
                                0.05)
    baseline[i, :] = new_state
    last_state = new_state

gn_traj = np.zeros((maxframe+1, 4))
gn_traj[0, :] = start_state
last_state = torch.tensor([start_state])
action = torch.tensor([action])
for i in range(1, maxframe+1):
    init_graph_features(G1, graph_feat_size, node_feat_size,
                        edge_feat_size, cuda=True, bs=1)
    load_graph_features(G1, action, last_state.clone(), None, bs=1,
                        noise=0, std=None)
    gn.eval()
    with torch.no_grad():
        G_out = gn(in_normalizer.normalize(G1))

    for node in G_out.nodes():
        gn_traj[i, [node, node+2]] = G_out.nodes[node]['feat'].cpu().detach()\
            .numpy()
    gn_traj[i] += last_state.numpy()[0]
    last_state = torch.tensor([gn_traj[i]])

plt.subplot(121)
plt.plot(range(maxframe+1), baseline[:, 0], range(maxframe+1), gn_traj[:, 0])
plt.plot(range(maxframe+1), baseline[:, 1], range(maxframe+1), gn_traj[:, 1])
plt.legend(['gt_a1', 'pred_a1', 'gt_a2', 'pred_a2'])
plt.subplot(122)
plt.plot(range(maxframe+1), baseline[:, 2], range(maxframe+1), gn_traj[:, 2])
plt.plot(range(maxframe+1), baseline[:, 3], range(maxframe+1), gn_traj[:, 3])
plt.legend(['gt_v1', 'pred_v1', 'gt_v2', 'pred_v2'])

plt.show()