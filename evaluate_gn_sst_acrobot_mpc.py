from sparse_rrt.systems import Acrobot
from gn_models import init_graph_features, FFGN
from utils import load_graph_features_sst_acrobot\
    as load_graph_features
import networkx as nx
import numpy as np
import torch
from mpc.mpc import QuadCost, GradMethods
from mpc import mpc
from matplotlib import pyplot as plt
from torch.autograd import Variable
from tqdm import tqdm

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
        action = control
        delta_state = torch.zeros_like(start_state)
        last_state = start_state.clone()
        for i in range(num_steps):
            init_graph_features(self.G1, self.graph_feat_size,
                                self.node_feat_size,
                                self.edge_feat_size, cuda=True, bs=1)
            load_graph_features(self.G1, action, last_state
                                .clone(), None, bs=1,
                                noise=0, std=None)
            self.gn.eval()
            # with torch.no_grad():
            G_out = self.gn(self.in_normalizer.normalize(self.G1))
            for node in G_out.nodes():
                delta_state[0, [node, node+2]] = G_out.nodes[node]['feat'][0].cpu()
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


class Acrobot_GN_MPC(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Acrobot_GN()
        self.dt = 0.05
        self.n_state = 4
        self.n_ctrl = 1
        self.lower, self.upper = self.model.MIN_TORQUE, self.model.MAX_TORQUE
        self.mpc_eps = 1e-3
        self.linesearch_decay = 0.2
        self.max_linesearch_iter = 5

    def forward(self, x, u):
        return self.model.propagate(x, u, 1, 0.05)

class AcrobotTorch(Acrobot):
    '''
    Two joints pendulum that is activated in the second joint (Acrobot)
    '''
    STATE_THETA_1, STATE_THETA_2, STATE_V_1, STATE_V_2 = 0, 1, 2, 3
    MIN_V_1, MAX_V_1 = -6., 6.
    MIN_V_2, MAX_V_2 = -6., 6.
    MIN_TORQUE, MAX_TORQUE = -8., 8.

    MIN_ANGLE, MAX_ANGLE = -np.pi, np.pi

    LENGTH = 20.
    m = 1.0
    lc = 0.5
    lc2 = 0.25
    l2 = 1.
    I1 = 0.2
    I2 = 1.0
    l = 1.0
    g = 9.81

    def propagate(self, start_state, control, num_steps, integration_step):
        last_state = start_state.clone()
        derv = integration_step * self._compute_derivatives(last_state, control)
        state = start_state.clone() + derv
        if state[0] < -np.pi:
            state[0] += 2*np.pi
        elif state[0] > np.pi:
            state[0] -= 2 * np.pi
        if state[1] < -np.pi:
            state[1] += 2*np.pi
        elif state[1] > np.pi:
            state[1] -= 2 * np.pi
        state_ = state.clone()
        state_[2:] = torch.clamp(
            state[2:],
            min=self.MIN_V_1,
            max=self.MAX_V_1)
        return state_

    def visualize_point(self, state):
        x2 = self.LENGTH * np.cos(state[self.STATE_THETA_1] - np.pi / 2) + self.LENGTH * np.cos(state[self.STATE_THETA_1] + state[self.STATE_THETA_2] - np.pi/2)
        y2 = self.LENGTH * np.sin(state[self.STATE_THETA_1] - np.pi / 2) + self.LENGTH * np.sin(state[self.STATE_THETA_1] + state[self.STATE_THETA_2] - np.pi / 2)
        x1 = self.LENGTH * np.cos(state[self.STATE_THETA_1] - np.pi / 2)
        y1 = self.LENGTH * np.sin(state[self.STATE_THETA_1] - np.pi / 2)
        # x2 = (x + 2 * self.LENGTH) / (4 * self.LENGTH)
        # y2 = (y + 2 * self.LENGTH) / (4 * self.LENGTH)
        return x1, y1, x2, y2

    def _compute_derivatives(self, state, control):
        '''
        Port of the cpp implementation for computing state space derivatives
        '''
        theta2 = state[self.STATE_THETA_2]
        theta1 = state[self.STATE_THETA_1] - np.pi/2
        theta1dot = state[self.STATE_V_1]
        theta2dot = state[self.STATE_V_2]
        _tau = control[0]
        m = self.m
        l2 = self.l2
        lc2 = self.lc2
        l = self.l
        lc = self.lc
        I1 = self.I1
        I2 = self.I2

        d11 = m * lc2 + m * (l2 + lc2 + 2 * l * lc * torch.cos(theta2)) + I1 + I2
        d22 = m * lc2 + I2
        d12 = m * (lc2 + l * lc * torch.cos(theta2)) + I2
        d21 = d12

        c1 = -m * l * lc * theta2dot * theta2dot * torch.sin(theta2) - (2 * m * l * lc * theta1dot * theta2dot * torch.sin(theta2))
        c2 = m * l * lc * theta1dot * theta1dot * torch.sin(theta2)
        g1 = (m * lc + m * l) * self.g * torch.cos(theta1) + (m * lc * self.g * torch.cos(theta1 + theta2))
        g2 = m * lc * self.g * torch.cos(theta1 + theta2)

        deriv = state.clone()
        deriv[self.STATE_THETA_1] = theta1dot
        deriv[self.STATE_THETA_2] = theta2dot

        u2 = _tau - 1 * .1 * theta2dot
        u1 = -1 * .1 * theta1dot
        theta1dot_dot = (d22 * (u1 - c1 - g1) - d12 * (u2 - c2 - g2)) / (d11 * d22 - d12 * d21)
        theta2dot_dot = (d11 * (u2 - c2 - g2) - d21 * (u1 - c1 - g1)) / (d11 * d22 - d12 * d21)
        deriv[self.STATE_V_1] = theta1dot_dot
        deriv[self.STATE_V_2] = theta2dot_dot
        return deriv

    def get_state_bounds(self):
        return [(self.MIN_ANGLE, self.MAX_ANGLE),
                (self.MIN_ANGLE, self.MAX_ANGLE),
                (self.MIN_V_1, self.MAX_V_1),
                (self.MIN_V_2, self.MAX_V_2)]

    def get_control_bounds(self):
        return [(self.MIN_TORQUE, self.MAX_TORQUE)]

    def distance_computer(self):
        return _sst_module.TwoLinkAcrobotDistance()


class Acrobot_MPC(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AcrobotTorch()
        self.dt = 1e-2
        self.n_state = 4
        self.n_ctrl = 1
        self.lower, self.upper = self.model.MIN_TORQUE, self.model.MAX_TORQUE
        self.mpc_eps = 1e-2
        self.linesearch_decay = 0.1
        self.max_linesearch_iter = 50

    def forward(self, x, u):
        return self.model.propagate(x.squeeze(), u, 1, self.dt).unsqueeze(0)
        # return self.model.propagate(x.squeeze(), u, 1, self.dt).unsqueeze(0) - x


if __name__ == '__main__':
    n_batch, T, mpc_T = 1, 1, 50
    dx = Acrobot_MPC()
    dx.eval()
    # dx = Acrobot_GN_MPC()

    def uniform(shape, low, high):
        r = high-low
        return torch.rand(shape)*r+low
    path = np.array([[ 0.        ,  0.        ,  0.        ,  0.        ],
                     [ 0.21286178, -0.46746978,  0.65104401, -1.5594405 ],
                     [-0.17452348,  0.14667169, -2.13591268,  4.06777428],
                     [-0.75232308,  1.50979792,  0.52563659,  0.49861793],
                     [ 0.25864789,  0.43957809,  3.17068376, -4.99579074],
                     [ 1.40715469, -1.90850489,  0.00868525, -2.59786582],
                     [ 0.22851573, -1.8938895 , -4.33178308,  3.29008524],
                     [-1.84396495,  0.75659586, -2.69265888,  5.05459236],
                     [-2.20080594,  2.6829071 ,  1.88980842,  2.55476466],
                     [-0.09057802,  2.82889481,  6.        , -2.20247192],
                     [ 2.2855393 ,  1.2390738 ,  2.52924645, -3.0075712 ],
                     [-3.02683454, -0.23204286,  2.12004888, -3.91888116]])

    torch.manual_seed(0)
    th1 = uniform(n_batch, -(1/2)*np.pi, (1/2)*np.pi)*0.05
    th2 = uniform(n_batch, -(1/2)*np.pi, (1/2)*np.pi)*0.05
    thdot1 = uniform(n_batch, -1., 1.)*0
    thdot2 = uniform(n_batch, -1., 1.)*0

    u_init = None

    # goal_weights = torch.Tensor((10., 10., 0., 0.))
    # goal_state = torch.Tensor((np.pi, 0., 0, 0.))
    goal_state = torch.Tensor(path[8])
    # goal_state = torch.Tensor((np.pi/4., 0., 0., 0.))
    # xinit = torch.tensor((-np.pi/2, 0, 0., 0.)).unsqueeze(0)
    # xinit = torch.Tensor((-0, -0, 0., 0.)).unsqueeze(0)
    xinit = torch.Tensor(path[7]).unsqueeze(0)
    # print(xinit)
    _, _, xg, yg = dx.model.visualize_point(goal_state.numpy())
    plt.scatter([xg], [yg], color='r', s=10)

    u = torch.zeros(mpc_T).uniform_() * 8 - 4
    u = Variable(u, requires_grad=True)

    optimizer = torch.optim.Adam([u], lr=3., betas=(0.9, 0.999))
    it_max = 10000

    for it in range(it_max):
        x = xinit.clone()
        states = [x]
        optimizer.zero_grad()
        for t in range(mpc_T):
            with torch.set_grad_enabled(True):
                states.append(dx(states[-1].clone(), [u[t]]).clone())
        # torch.cat(cost, dim=0).sum().backward(retain_graph=True)
        terminal = states[-1]
        dis = torch.abs(terminal - goal_state.unsqueeze(0))
        d = dis.clone()
        d[:, :2] = torch.min(dis[:, :2], 2*np.pi - dis[:, :2])
        loss = (d**2).sum()
        loss.backward(retain_graph=True)
        optimizer.step()
        print(loss.item())
        if loss.item() < dx.mpc_eps:
            break

    x = xinit.clone()
    for control in u:
        with torch.no_grad():
            x = dx(x, [control])
            x1, y1, x2, y2 = dx.model.visualize_point(x.detach().cpu().numpy()[0])
            plt.plot([0]+[x1]+[x2], [0]+[y1]+[y2], color='gray')
            plt.scatter([x2], [y2], color='yellow', s=10)

    x1, y1, x2, y2 = dx.model.visualize_point(xinit.detach().cpu().numpy()[0])
    plt.plot([0]+[x1]+[x2], [0]+[y1]+[y2], color='blue')
    # x1, y1, x2, y2 = dx.model.visualize_point(xinit.numpy()[0])
    # plt.scatter([x2], [y2], color='g', s=10)

    x1, y1, x2, y2 = dx.model.visualize_point(states[-1].detach().cpu().numpy()[0])
    plt.plot([0]+[x1]+[x2], [0]+[y1]+[y2], color='black')
    plt.scatter([x2], [y2], color='orange', s=10)
    plt.axis('equal')
    plt.show()
