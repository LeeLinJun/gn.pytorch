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
        state = start_state
        for i in range(num_steps):
            state += integration_step * self._compute_derivatives(state, control)

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
        self.dt = 5e-2
        self.n_state = 4
        self.n_ctrl = 1
        self.lower, self.upper = self.model.MIN_TORQUE, self.model.MAX_TORQUE
        self.mpc_eps = 1e-3
        self.linesearch_decay = 0.2
        self.max_linesearch_iter = 5

    def forward(self, x, u):
        return self.model.propagate(x.squeeze(), u, 1, self.dt).unsqueeze(0)


if __name__ == '__main__':
    n_batch, T, mpc_T = 1, 20, 20
    # dx = Acrobot_MPC()
    dx = Acrobot_GN_MPC()

    def uniform(shape, low, high):
        r = high-low
        return torch.rand(shape)*r+low
    path = np.array([[0., 0., 0., 0.],
                     [0.15517462, -0.3698205, 0.11219044, -0.47243161],
                     [-0.20895943, 0.22132772, -1.1241821, 2.19049586],
                     [-0.10750574, 0.50542, 1.47872063, -1.74839844],
                     [0.62824243, -0.59365016, 0.20789198, -0.7827772],
                     [-0.56960533, 0.96463903, -2.49284853, 3.34120039],
                     [-1.13182336, 1.85553684, 1.05989811, -0.37408504],
                     [1.25632005, -1.04712924, 3.37493111, -4.28736475],
                     [1.94917096, -2.31124249, -1.25351372, -0.62326882],
                     [-1.36201506, 0.47373986, -5.22150959, 5.38225952],
                     [-2.77032012, 1.41657443, -0.12287768, -0.3727394],
                     [-2.32652658, 1.37776851, 2.226437, -0.40219697],
                     [0.92427265, -1.08004294, 4.6829297, -2.72950627],
                     [1.54028254, 0.15374353, -3.0370493, 6.]])

    # torch.manual_seed(0)
    th1 = uniform(n_batch, -(1/2)*np.pi, (1/2)*np.pi)*0.1
    th2 = uniform(n_batch, -(1/2)*np.pi, (1/2)*np.pi)*0.1
    thdot1 = uniform(n_batch, -1., 1.)*0
    thdot2 = uniform(n_batch, -1., 1.)*0

    u_init = None

    goal_weights = torch.Tensor((1., 1., 0., 0.))
    # goal_state = torch.Tensor((np.pi, 0., 0, 0.))
    goal_state = torch.Tensor(path[3])
    # goal_state = torch.Tensor((np.pi/4., 0., 0., 0.))
    # xinit = torch.stack((th1, thdot1, th2, thdot2), dim=1)
    # xinit = torch.Tensor((-0, -0, 0., 0.)).unsqueeze(0)
    xinit = torch.Tensor(path[2]).unsqueeze(0)
    print(xinit)

    state = x = xinit.clone()
    ctrl_penalty = 1e-2
    q = torch.cat((
        goal_weights,
        ctrl_penalty*torch.ones(1)
    ))
    px = -torch.sqrt(goal_weights)*goal_state
    p = torch.cat((px, torch.zeros(dx.n_ctrl)))
    Q = torch.diag(q).unsqueeze(0).unsqueeze(0).repeat(
        mpc_T, n_batch, 1, 1
    )
    p = p.unsqueeze(0).repeat(mpc_T, n_batch, 1)

    _, _, xg, yg = dx.model.visualize_point(goal_state.numpy())
    plt.scatter([xg], [yg], color='r', s=10)

    for i in range(T):
        print(i)
        x = state.clone()
        nominal_states, nominal_actions, nominal_objs = mpc.MPC(dx.n_state, dx.n_ctrl, mpc_T,
                                                                u_init=u_init,
                                                                u_lower=dx.lower, u_upper=dx.upper,
                                                                lqr_iter=500,
                                                                verbose=0,
                                                                exit_unconverged=False,
                                                                detach_unconverged=False,
                                                                linesearch_decay=dx.linesearch_decay,
                                                                max_linesearch_iter=dx.max_linesearch_iter,
                                                                grad_method=GradMethods.AUTO_DIFF,
                                                                eps=1e-9,
                                                                ).cuda()(x, QuadCost(Q, p), dx) 
        x1, y1, x2, y2 = dx.model.visualize_point(state.detach().cpu().numpy()[0])
        print(nominal_objs)
        plt.plot([0]+[x1]+[x2], [0]+[y1]+[y2], color='gray')
        # if (x2 - xg)**2 + (y2 - yg)**2 < 0.2:
        #     break

        next_action = nominal_actions[0]
        u_init = torch.cat((nominal_actions[1:], torch.zeros(1, n_batch, dx.n_ctrl)), dim=0)
        u_init[-2] = u_init[-3]
        with torch.no_grad():
            state = dx(state, next_action)
        # print(state)
        # plt.scatter([x1, x2], [y1, y2], color='orange', s=10)
        # print(state)

    # for state in nominal_states:
    #     x1, y1, x2, y2 = dx.model.visualize_point(state.detach().cpu().numpy()[0])
    #     plt.plot([0]+[x1]+[x2], [0]+[y1]+[y2], color='gray')
    #     plt.scatter([x1, x2], [y1, y2], color='orange', s=10)

    x1, y1, x2, y2 = dx.model.visualize_point(xinit.detach().cpu().numpy()[0])
    plt.plot([0]+[x1]+[x2], [0]+[y1]+[y2], color='blue')
    x1, y1, x2, y2 = dx.model.visualize_point(xinit.numpy()[0])
    plt.scatter([x2], [y2], color='g', s=10)

    x1, y1, x2, y2 = dx.model.visualize_point(state.detach().cpu().numpy()[0])
    plt.plot([0]+[x1]+[x2], [0]+[y1]+[y2], color='black')
    plt.scatter([x2], [y2], color='orange', s=10)
    plt.axis('equal')
    plt.show()
