from sparse_rrt.systems import Acrobot
from gn_models import init_graph_features, FFGN
from utils import load_graph_features_sst_acrobot\
    as load_graph_features
import networkx as nx
import numpy as np
import torch
from sparse_rrt.planners import SST
from matplotlib import pyplot as plt

class Acrobot_GN(Acrobot):
    '''
    '''
    STATE_THETA_1, STATE_THETA_2, STATE_V_1, STATE_V_2 = 0, 1, 2, 3
    MIN_V_1, MAX_V_1 = -6., 6.
    MIN_V_2, MAX_V_2 = -6., 6.
    MIN_TORQUE, MAX_TORQUE = -4., 4.

    MIN_ANGLE, MAX_ANGLE = -np.pi, np.pi

    LENGTH = 20.
    m = 1.0
    lc = 0.5
    lc2 = 0.25
    l2 = 1.
    I1 = 0.2
    I2 = 1.0
    g = 9.81

    def __init__(self):
        super(Acrobot, self).__init__()
        self.G1 = nx.path_graph(2).to_directed()
        self.node_feat_size = 2
        self.edge_feat_size = 3
        self.graph_feat_size = 10
        self.gn = FFGN(self.graph_feat_size, self.node_feat_size,
                       self.edge_feat_size).cuda()
        self.gn.load_state_dict(torch.load('model0.05.pth'))
        normalizers = torch.load('normalized/acrobot0.05.pth')
        self.in_normalizer = normalizers['in_normalizer']
        self.out_normalizer = normalizers['out_normalizer']

    def propagate(self, start_state, control, num_steps, integration_step):
        '''
        Integrate system dynamics
        :param start_state: numpy array with the start state for the integration
        :param control: numpy array with constant controls to be applied during integration
        :param num_steps: number of steps to integrate
        :param integration_step: dt of integration
        :return: new state of the system
        '''
        action = torch.from_numpy(np.array([control]))
        delta_state = np.zeros_like(start_state)
        last_state = start_state.copy()
        for i in range(num_steps):
            init_graph_features(self.G1, self.graph_feat_size,
                                self.node_feat_size,
                                self.edge_feat_size, cuda=True, bs=1)
            load_graph_features(self.G1, action, torch.tensor([last_state])
                                .clone(), None, bs=1,
                                noise=0, std=None)
            self.gn.eval()
            with torch.no_grad():
                G_out = self.gn(self.in_normalizer.normalize(self.G1))
            for node in G_out.nodes():
                delta_state[[node, node+2]] = G_out.nodes[node]['feat'].cpu()\
                    .detach().numpy()[0]
            last_state += delta_state
        return last_state

    def visualize_point(self, state):
        x2 = self.LENGTH * np.cos(state[self.STATE_THETA_1] - np.pi / 2) + self.LENGTH * np.cos(state[self.STATE_THETA_1] + state[self.STATE_THETA_2] - np.pi/2)
        y2 = self.LENGTH * np.sin(state[self.STATE_THETA_1] - np.pi / 2) + self.LENGTH * np.sin(state[self.STATE_THETA_1] + state[self.STATE_THETA_2] - np.pi / 2)
        x1 = self.LENGTH * np.cos(state[self.STATE_THETA_1] - np.pi / 2)
        y1 = self.LENGTH * np.sin(state[self.STATE_THETA_1] - np.pi / 2)
        # x2 = (x + 2 * self.LENGTH) / (4 * self.LENGTH)
        # y2 = (y + 2 * self.LENGTH) / (4 * self.LENGTH)
        return x1, y1, x2, y2

    def get_state_bounds(self):
        return [(self.MIN_ANGLE, self.MAX_ANGLE),
                (self.MIN_ANGLE, self.MAX_ANGLE),
                (self.MIN_V_1, self.MAX_V_1),
                (self.MIN_V_2, self.MAX_V_2)]

    def get_control_bounds(self):
        return [(self.MIN_TORQUE, self.MAX_TORQUE)]


if __name__ == '__main__':
    model_l = Acrobot_GN()  
    model = Acrobot()  #
    system = Acrobot()

    start_state = np.array([0., 0., 0., 0.])
    goal_state = np.array([3.14, 0, 0., 0.])
    curr_state = start_state

    while True:
        print("Current state:", curr_state)
        planner = SST(
            state_bounds=model.get_state_bounds(),
            control_bounds=model.get_control_bounds(),
            distance=model.distance_computer(),
            start_state=curr_state,
            goal_state=goal_state,
            goal_radius=2.0,
            random_seed=0,
            sst_delta_near=1.,
            sst_delta_drain=0.5,
        )
        for iteration in range(20000):
            planner.step(model, 10, 15, 0.05)
            if iteration % 100 == 0:
                solution = planner.get_solution()
                print("Solution: %s, Number of nodes: %s" %
                      (solution, planner.get_number_of_nodes()))
                if solution is not None:
                    break
        controls = solution[1]  # path,control,cost
        # for i in range(len(controls)):
        #     curr_state = model.propagate(curr_state, controls[i],
        #                                  1,
        #                                  0.05)
        #     if np.linalg.norm(curr_state-solution[0][i]) > 2:
        #         print("mismatch at %d steps" % i)
        #         break
        for state in solution[0]:
            x1, y1, x2, y2 = model_l.visualize_point(state)
            plt.plot([0]+[x1]+[x2], [0]+[y1]+[y2], color='gray')
            plt.scatter(x2, y2, color='orange', s=10)

        plt.show()

        if np.linalg.norm(curr_state-goal_state) < 2.0:
            print("sucesses!")
            break
