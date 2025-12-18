import numpy as np
from matplotlib import pyplot as plt
import os
import time
from sklearn.cluster import KMeans
import copy
# import rospy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.transform import Rotation as sciR

import torch_rbf as rbf  # reference: https://github.com/JeremyLinux/PyTorch-Radial-Basis-Function-Layer

from utils.data_augmentation import dataRandomTransformWithLength
from utils.state_index import I

import json

from torch_geometric.data import Data, Batch
import torch_geometric as pyg
from torch_geometric.nn import radius_graph
from GN_model_DLO import MySubEdge, MySubNode
from torch_geometric.nn import global_mean_pool

import argparse

parser = argparse.ArgumentParser(description="DLO Parser")


def load_args_from_file(
        filepath=r"/home/jessica/shape_control_DLO_2/ws_dlo/src/dlo_system_pkg/config/config.json"
        ):
    with open(filepath, 'r') as file:
        args_dict = json.load(file)
    # Convert dictionary back to Namespace
    args = argparse.Namespace(**args_dict)
    return args


args = load_args_from_file()
# print(args)

params_online_window_time = 2  # unit: second
params_online_max_valid_fps_vel = 0.3
params_online_fps_vel_thres = 0.01
params_online_min_valid_fps_vel = 0.00
params_update_if_window_full = False


class Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


args_GN = Args(
    lr=1e-4
)


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------
# seperate the x, s, v, x_dot from the state vector (dim=117)
class NNDataset(Dataset):
    def __init__(self, state):
        self.data_num = state.shape[0]
        self.length = state[:, I.length_idx]
        self.state_input = state[:, I.state_input_idx]
        self.fps_vel = state[:, I.fps_vel_idx]
        self.ends_vel = state[:, I.ends_vel_idx]

    def __getitem__(self, index):
        return self.length[index], self.state_input[index], self.fps_vel[index], self.ends_vel[index]

    def __len__(self):
        return self.data_num


def mask_feature_points_batch(state_input_tensor, p_mask=0.25, numFPs=10):
    """
    Randomly hide a contiguous block of feature points in some samples.

    Inputs:
        state_input_tensor: (B, 3*numFPs + 14) torch.Tensor
        p_mask: probability a given sample is masked
        numFPs: number of feature points (e.g. 10)

    Returns:
        state_input_masked: same shape, with some FP positions zeroed.
    """
    B = state_input_tensor.shape[0]
    state_out = state_input_tensor.clone()

    # indices of the FP position part in the flattened state_input
    # fps_pos is stored first: 0 : 3*numFPs
    for i in range(B):
        if np.random.rand() < p_mask:
            # choose k_hidden in {1,2,3,4}
            k_hidden = np.random.randint(1, 5)  # upper bound exclusive → [1,4]
            # choose start index so that block fits inside [0, numFPs-1]
            start = np.random.randint(0, numFPs - k_hidden + 1)
            end = start + k_hidden  # not inclusive

            # zero positions of those FPs (3 coords per FP)
            for fp_idx in range(start, end):
                s = 3 * fp_idx
                e = 3 * (fp_idx + 1)
                state_out[i, s:e] = 0.0   # xyz of FP fp_idx

    return state_out

def mask_feature_points_batch_2(state_input_tensor, p_mask=0.25, numFPs=10):
    """
    Randomly hide a contiguous block of feature points in some samples.

    Returns:
        state_input_masked: (B, 3*numFPs+14)
        fp_visible_mask:    (B, numFPs)  1=visible, 0=hidden
    """
    B = state_input_tensor.shape[0]
    state_out = state_input_tensor.clone()
    fp_visible_mask = torch.ones(B, numFPs, dtype=torch.float32)

    for i in range(B):
        if np.random.rand() < p_mask:
            k_hidden = np.random.randint(1, 5)
            start = np.random.randint(0, numFPs - k_hidden + 1)
            end = start + k_hidden

            for fp_idx in range(start, end):
                s = 3 * fp_idx
                e = 3 * (fp_idx + 1)
                state_out[i, s:e] = 0.0
                fp_visible_mask[i, fp_idx] = 0.0

    return state_out, fp_visible_mask

def mask_feature_points_batch_noncontig(state_input_tensor, p_mask=0.25, numFPs=10):
    """
    Randomly hide a non-contiguous set of feature points in some samples.

    Inputs:
        state_input_tensor: (B, 3*numFPs + 14) torch.Tensor
        p_mask: probability a given sample is masked
        numFPs: number of feature points (e.g. 10)

    Returns:
        state_input_masked: same shape, with some FP positions zeroed.
    """
    B = state_input_tensor.shape[0]
    state_out = state_input_tensor.clone()

    for i in range(B):
        if np.random.rand() < p_mask:
            # choose k_hidden in {1,2,3,4}
            k_hidden = np.random.randint(1, 5)  # 1,2,3,4

            # randomly choose k_hidden distinct FP indices (non-contiguous allowed)
            hidden_indices = np.random.choice(numFPs, size=k_hidden, replace=False)

            # zero positions of those FPs (3 coords per FP)
            for fp_idx in hidden_indices:
                s = 3 * fp_idx
                e = 3 * (fp_idx + 1)
                state_out[i, s:e] = 0.0   # xyz of FP fp_idx

    return state_out

def mask_feature_points_batch_noncontig_2(state_input_tensor, p_mask=0.25, numFPs=10):
    """
    Randomly hide a non-contiguous set of feature points.
    """
    B = state_input_tensor.shape[0]
    state_out = state_input_tensor.clone()
    fp_visible_mask = torch.ones(B, numFPs, dtype=torch.float32)

    for i in range(B):
        if np.random.rand() < p_mask:
            k_hidden = np.random.randint(1, 5)
            hidden_indices = np.random.choice(numFPs, size=k_hidden, replace=False)

            for fp_idx in hidden_indices:
                s = 3 * fp_idx
                e = 3 * (fp_idx + 1)
                state_out[i, s:e] = 0.0
                fp_visible_mask[i, fp_idx] = 0.0

    return state_out, fp_visible_mask

def mask_feature_points_batch_combined(state_input_tensor, p_mask=0.25, numFPs=10):
    """
    Combined masking: randomly chooses contiguous OR non-contiguous FP hiding.
    
    Inputs:
        state_input_tensor: (B, 3*numFPs + 14) torch.Tensor
        p_mask: probability a given sample is masked (either type)
        numFPs: number of feature points (e.g. 10)
    
    Returns:
        state_input_masked: same shape, with some FP positions zeroed.
    """
    B = state_input_tensor.shape[0]
    state_out = state_input_tensor.clone()

    for i in range(B):
        if np.random.rand() < p_mask:
            # 50% chance: contiguous block, 50% chance: non-contiguous
            if np.random.rand() < 0.5:
                # CONTIGUOUS masking
                k_hidden = np.random.randint(1, 5)  # [1,4]
                start = np.random.randint(0, numFPs - k_hidden + 1)
                end = start + k_hidden
                for fp_idx in range(start, end):
                    s = 3 * fp_idx
                    e = 3 * (fp_idx + 1)
                    state_out[i, s:e] = 0.0
            else:
                # NON-CONTIGUOUS masking
                k_hidden = np.random.randint(1, 5)  # [1,4]
                hidden_indices = np.random.choice(numFPs, size=k_hidden, replace=False)
                for fp_idx in hidden_indices:
                    s = 3 * fp_idx
                    e = 3 * (fp_idx + 1)
                    state_out[i, s:e] = 0.0

    return state_out

def mask_feature_points_batch_combined_2(state_input_tensor, p_mask=0.25, numFPs=10):
    """
    Combined masking: randomly contiguous OR non-contiguous.
    """
    B = state_input_tensor.shape[0]
    state_out = state_input_tensor.clone()
    fp_visible_mask = torch.ones(B, numFPs, dtype=torch.float32)

    for i in range(B):
        if np.random.rand() < p_mask:
            if np.random.rand() < 0.5:
                # contiguous
                k_hidden = np.random.randint(1, 5)
                start = np.random.randint(0, numFPs - k_hidden + 1)
                end = start + k_hidden
                hidden_indices = range(start, end)
            else:
                # non-contiguous
                k_hidden = np.random.randint(1, 5)
                hidden_indices = np.random.choice(numFPs, size=k_hidden, replace=False)

            for fp_idx in hidden_indices:
                s = 3 * fp_idx
                e = 3 * (fp_idx + 1)
                state_out[i, s:e] = 0.0
                fp_visible_mask[i, fp_idx] = 0.0

    return state_out, fp_visible_mask

# def get_graph_data(state_input, length, numFPs):
#     # state_input dim is [1,44]
#     left_end_pos = state_input[:, 3 * numFPs: 3 * numFPs + 3]
#     left_end_quat = state_input[:, 3 * numFPs + 3: 3 * numFPs + 7]
#     right_end_pos = state_input[:, 3 * numFPs + 7: 3 * numFPs + 10]
#     right_end_quat = state_input[:, 3 * numFPs + 10: 3 * numFPs + 14]
#     fps_pos = state_input[:, 0: 3 * numFPs].reshape(numFPs, 3)

#     #Coordinate Axis Origin Augmentation
#     # Choose origin at FP1 (feature point index 0)
#     origin = fps_pos[0:1, :]  # shape [1, 3]

#      # Shift everything so origin is FP1
#     fps_pos_rel       = fps_pos       - origin          # [numFPs, 3]
#     left_end_pos_rel  = left_end_pos  - origin          # [1, 3]
#     right_end_pos_rel = right_end_pos - origin          # [1, 3]

#     # node_input dim is 7
#     # [x,y,z,angle pos]
#     # angle_pos is [0,0,0,0] for fps, and [q1, q2, q3, q4] for ends
#     left_end_state = torch.concat((left_end_pos_rel, left_end_quat), dim=-1)
#     right_end_state = torch.concat((right_end_pos_rel, right_end_quat), dim=-1)
#     fps_state = torch.concat((fps_pos_rel, torch.zeros(numFPs, 4)), dim=-1)

#     # left_end_state = torch.concat((left_end_pos, left_end_quat), dim=-1)
#     # right_end_state = torch.concat((right_end_pos, right_end_quat), dim=-1)
#     # fps_state = torch.concat((fps_pos, torch.zeros(numFPs, 4)), dim=-1)

#     node_input = torch.concat((left_end_state, fps_state, right_end_state), dim=0)

#     # dim = [num_fps + 2, 3]
#     all_pos_xyz = node_input[:, :3]

#     # edge_index depends on redius
#     # edge_index = radius_graph(all_pos_xyz, r=length / 3, loop=False)
#     edge_index = radius_graph(all_pos_xyz, r=length, loop=False)

#     # edge feature input, relative distance
#     edge_input = all_pos_xyz[edge_index[0]] - all_pos_xyz[edge_index[1]]

#     # return the graph with features
#     graph = pyg.data.Data(
#         x=node_input,
#         edge_index=edge_index,
#         edge_attr=edge_input,
#         all_pos_xyz=all_pos_xyz
#     )

#     return graph

def get_graph_data(state_input, length, fp_visible_mask, numFPs):
    # state_input dim is [1,44]
    left_end_pos = state_input[:, 3 * numFPs: 3 * numFPs + 3]
    left_end_quat = state_input[:, 3 * numFPs + 3: 3 * numFPs + 7]
    right_end_pos = state_input[:, 3 * numFPs + 7: 3 * numFPs + 10]
    right_end_quat = state_input[:, 3 * numFPs + 10: 3 * numFPs + 14]
    fps_pos = state_input[:, 0: 3 * numFPs].reshape(numFPs, 3)

    origin = fps_pos[0:1, :]
    fps_pos_rel       = fps_pos       - origin
    left_end_pos_rel  = left_end_pos  - origin
    right_end_pos_rel = right_end_pos - origin

    left_end_state = torch.concat((left_end_pos_rel, left_end_quat), dim=-1)
    right_end_state = torch.concat((right_end_pos_rel, right_end_quat), dim=-1)
    fps_state = torch.concat((fps_pos_rel, torch.zeros(numFPs, 4)), dim=-1)  # (numFPs,7)

    # Build mask channel: 1 for visible, 0 for hidden
    # Ends are always visible → mask=1
    mask_left = torch.ones(1, 1, dtype=torch.float32)
    mask_fp   = fp_visible_mask.reshape(numFPs, 1).to(torch.float32)
    mask_right= torch.ones(1, 1, dtype=torch.float32)

    mask_channel = torch.cat([mask_left, mask_fp, mask_right], dim=0)  # (numFPs+2,1)

    node_input = torch.concat((left_end_state, fps_state, right_end_state), dim=0)  # (numFPs+2,7)
    node_input = torch.cat([node_input, mask_channel], dim=1)  # (numFPs+2, 8)  ← extra mask bit

    all_pos_xyz = node_input[:, :3]
    edge_index = radius_graph(all_pos_xyz, r=length, loop=False)
    edge_input = all_pos_xyz[edge_index[0]] - all_pos_xyz[edge_index[1]]

    graph = pyg.data.Data(
        x=node_input,
        edge_index=edge_index,
        edge_attr=edge_input,
        all_pos_xyz=all_pos_xyz
    )

    return graph


# def get_graph_batch(state_input_tensor, length_tensor, numFPs=10):
#     # get a list of graph
#     return [get_graph_data(state_input_tensor[i:i + 1, :], length_tensor[i], numFPs) for i in
#             range(state_input_tensor.shape[0])]

# def get_graph_batch(state_input_tensor, length_tensor, numFPs=10):
#     # Ensure length_tensor is 1D of shape (B,)
#     if isinstance(length_tensor, torch.Tensor):
#         length_tensor = length_tensor.view(-1)  # flatten: (B,*) -> (B,)

#     graphs = []
#     B = state_input_tensor.shape[0]

#     for i in range(B):
#         state_i = state_input_tensor[i:i+1, :]  # (1, feat_dim)

#         # length_tensor[i] should now be scalar-like
#         length_i = length_tensor[i]

#         if isinstance(length_i, torch.Tensor):
#             r_i = float(length_i.item()) / 3.0
#         else:
#             r_i = float(length_i) / 3.0

#         graphs.append(get_graph_data(state_i, r_i, numFPs))

#     return graphs

def get_graph_batch(state_input_tensor, length_tensor, fp_visible_mask, numFPs=10):
    # length_tensor: (B,)
    # fp_visible_mask: (B, numFPs)
    if isinstance(length_tensor, torch.Tensor):
        length_tensor = length_tensor.view(-1)

    graphs = []
    B = state_input_tensor.shape[0]

    for i in range(B):
        state_i = state_input_tensor[i:i+1, :]
        length_i = length_tensor[i]
        mask_i = fp_visible_mask[i]          # (numFPs,)

        if isinstance(length_i, torch.Tensor):
            r_i = float(length_i.item()) / 3.0
        else:
            r_i = float(length_i) / 3.0

        graphs.append(get_graph_data(state_i, r_i, mask_i, numFPs))

    return graphs




# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------
# class Net_J(nn.Module):
#     def __init__(self, nFPs, bTrainMuBeta, num_hidden_unit):
#         super(Net_J, self).__init__()
#         self.nFPs = nFPs
#         self.numHidden = num_hidden_unit
#         lw = [3*self.nFPs + 3+4+4, self.numHidden, (nFPs * 3) * 12]
#         print("Network dimension", lw)
#         basis_func = rbf.gaussian
#
#         self.fc1 = rbf.RBF(lw[0], lw[1], basis_func, bTrainMuBeta=bTrainMuBeta)
#         self.fc2 = nn.Linear(lw[1], lw[2], bias=False)
#
#
#     def forward(self, x):
#         theta = (self.fc1(x)).float()
#         output = (self.fc2(theta))
#         # print("output shape of RBF", output.shape)
#         # first 3 rows predict pos for feature 1, column 1
#         # first 36 rows for feature 1, all columns
#         # reshape follows the sequence of rows in J
#         # output dim = [batch_size, 360] 360 = num_FPs*12*3
#         output = torch.reshape(torch.reshape(output, (-1, self.nFPs, 12, 3)).transpose(2, 3), (-1, 3 * self.nFPs, 12)) # J: dimension: 30 * 12
#         return output
#
#     # use kmeans to calculate the initial value of mu and sigma in RBFN
#     def GetMuAndBetaByKMeans(self, full_data):
#         max_data_size = 600 * 60
#         if(full_data.shape[0] > max_data_size):
#             # randomly choose a subset of train data for kmeans
#             index = np.random.choice(np.arange(full_data.shape[0]), size=max_data_size, replace=False)
#             data = full_data[index, :]
#         else:
#             data = full_data
#
#         print("start kmeans ... ")
#         kmeans = KMeans(n_clusters=self.numHidden, n_init=2, max_iter=100).fit(data)
#         print("finish kmeans ... ")
#         nSamples = np.zeros((self.numHidden, ), dtype='float32')
#         variance = np.zeros((self.numHidden, ), dtype='float32')
#         for i, label in enumerate(kmeans.labels_):
#             variance[label] += np.linalg.norm(data[i, :] - kmeans.cluster_centers_[label, :])**2
#             nSamples[label] += 1
#         variance = variance / nSamples
#         sigma = np.sqrt(variance) * np.sqrt(2) * 10 #  mannually set initial value which is better for the following training
#
#         # error with invSigma = np.clip(invSigma, 0, 1)
#         invSigma = 1.0 / sigma
#         invSigma = np.clip(invSigma, 0, 1)
#         self.fc1.centres.data = torch.tensor(kmeans.cluster_centers_).to('cuda' if torch.cuda.is_available() else 'cpu')
#         self.fc1.sigmas.data = torch.tensor(invSigma).to('cuda' if torch.cuda.is_available() else 'cpu')

class MySimulator(torch.nn.Module):
    """Graph Network-based Simulators(GNS)"""

    def __init__(self):
        super().__init__()

        self.args = args_GN
        self.node_dim = 8 # 7
        self.edge_dim = 3
        self.hidden_dim = 256
        self.output_dim = 360  # size of J

        self.edge_prop_layer = torch.nn.Linear(350, 100, bias=False)
        self.node_prop_layer = torch.nn.Linear(200, 100, bias=False)

        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)

        subnet0_ee = MySubEdge('ee', self.edge_dim, self.hidden_dim, self.hidden_dim, 2, self.args)
        subnet1_ee = MySubEdge('ee', self.hidden_dim, self.hidden_dim, self.hidden_dim, 2, self.args)
        subnet2_ee = MySubEdge('ee', self.hidden_dim, self.hidden_dim, self.hidden_dim, 2, self.args)
        subnet3_en = MySubNode('en', self.node_dim, self.hidden_dim, self.hidden_dim, 2, self.args)
        subnet4_en = MySubNode('en', self.hidden_dim, self.hidden_dim, self.hidden_dim, 2, self.args)

        subnet5_pe = MySubEdge('pe', 3 * self.hidden_dim, self.hidden_dim, 2 * self.hidden_dim, 2, self.args)
        subnet6_pn = MySubNode('pn', 2 * self.hidden_dim, self.hidden_dim, self.hidden_dim, 2, self.args)

        subnet7_pe = MySubEdge('pe', 3 * self.hidden_dim, self.hidden_dim, 2 * self.hidden_dim, 2, self.args)
        subnet8_pn = MySubNode('pn', 2 * self.hidden_dim, self.hidden_dim, self.hidden_dim, 2, self.args)

        subnet9_pe = MySubEdge('pe', 3 * self.hidden_dim, self.hidden_dim, 2 * self.hidden_dim, 2, self.args)
        subnet10_pn = MySubNode('pn', 2 * self.hidden_dim, self.hidden_dim, self.hidden_dim, 2, self.args)

        # subnet11_dn = MySubNode('dn', 100, 100, 100, 2,self.args)

        # self.layers = torch.nn.ModuleList([MySubEdge('pe', 350,100, 200, 2), MySubNode(200, 100, 2), MySubNode(100, 100, 2)])
        # self.subnets= torch.nn.ModuleList([subnet0_ee, subnet1_ee, subnet2_ee, subnet3_en, subnet4_en, subnet5_pe, subnet6_pn, subnet5_pe, subnet6_pn, subnet5_pe, subnet6_pn, subnet11_dn])
        # self.subnets= torch.nn.ModuleList([subnet0_ee, subnet3_en, subnet5_pe, subnet6_pn])
        self.subnets = torch.nn.ModuleList(
            [subnet0_ee, subnet1_ee, subnet2_ee, subnet3_en, subnet4_en, subnet5_pe, subnet6_pn, subnet7_pe, subnet8_pn,
             subnet9_pe, subnet10_pn])

    def forward(self, data):
        node_feature = data.x
        edge_feature = data.edge_attr
        edge_index = data.edge_index
        batch = data.batch  # shape: [num_nodes], required for batching

        for subnet in self.subnets:
            if subnet.type == 'ee':
                node_out, edge_feature = subnet.forward(node_feature, node_feature, edge_index, edge_feature)
            elif subnet.type == 'en':
                node_out, node_feature = subnet.forward(node_feature)
            elif subnet.type == 'pe':
                node_out, edge_feature, input_to_node = subnet.forward(node_feature, node_feature, edge_index,
                                                                       edge_feature)
            elif subnet.type == 'pn':
                node_out, node_feature = subnet.forward(input_to_node)
            elif subnet.type == 'dn':
                node_out, _ = subnet.forward(node_feature)

        # Batch-aware graph-level readout
        graph_repr = global_mean_pool(node_feature, batch)  # shape: [num_graphs, feat_dim]
        output = self.output_layer(graph_repr)  # shape: [num_graphs, 360]

        return output


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------
class JacobianPredictor(object):
    #
    # numFPs = rospy.get_param("DLO/num_FPs")
    # projectDir = rospy.get_param("project_dir")
    # online_learning_rate = rospy.get_param("controller/online_learning/learning_rate")
    # lr_task_e = online_learning_rate
    # lr_approx_e = online_learning_rate * rospy.get_param("controller/online_learning/weight_ratio")
    # env = rospy.get_param("env/sim_or_real")
    # env_dim = rospy.get_param("env/dimension")
    # control_rate = rospy.get_param("ros_rate/env_rate")
    # online_update_rate = rospy.get_param("ros_rate/online_update_rate")

    numFPs = args.DLO_num_FPs
    projectDir = args.project_dir
    online_learning_rate = args.controller_online_learning_learning_rate
    lr_task_e = online_learning_rate
    lr_approx_e = online_learning_rate * args.controller_online_learning_weight_ratio
    env = args.env_sim_or_real
    env_dim = args.env_dimension
    control_rate = args.ros_rate_env_rate
    online_update_rate = args.ros_rate_online_update_rate

    # ------------------------------------------------------
    def __init__(self, num_hidden_unit=256):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.bTrainMuBeta = True

        self.model_J = MySimulator().to('cuda' if torch.cuda.is_available() else 'cpu')
        self.optimizer = torch.optim.Adam(self.model_J.parameters(), lr=1e-3)

        # for offline learning
        # torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99, last_epoch=-1, verbose=False)
        self.criterion = torch.nn.SmoothL1Loss(reduction='mean', beta=0.001)
        # for online learning
        # self.online_optimizer = torch.optim.SGD([{'params': self.model_J.fc2.parameters()}], lr=1.0/self.online_update_rate)
        self.mse_criterion = torch.nn.MSELoss(reduction='sum')

        # if rospy.get_param("learning/is_test"):
        if args.learning_is_test:
            # self.nnWeightDir = self.projectDir + 'ws_dlo/src/dlo_manipulation_pkg/models_test/rbfWeights/' + self.env_dim + '/'
            self.nnWeightDir = os.path.join(
                self.projectDir,
                'ws_dlo', 'src', 'dlo_manipulation_pkg', 'models_test',
                'rbfWeights', self.env_dim, ''
            )
        else:
            # self.nnWeightDir = self.projectDir + 'ws_dlo/src/dlo_manipulation_pkg/models/rbfWeights/' + self.env_dim + '/'
            self.nnWeightDir = os.path.join(
                self.projectDir,
                'ws_dlo', 'src', 'dlo_manipulation_pkg', 'models',
                'rbfWeights', self.env_dim, ''
            )
        # self.resultsDir = self.projectDir + 'results/' + self.env + '/'
        self.resultsDir = os.path.join(self.projectDir, 'results', self.env, '')
        # self.dataDir = self.projectDir +'data/'
        self.dataDir = os.path.join(self.projectDir, 'data', '')

        print('Directory')
        print(self.nnWeightDir)
        print(self.resultsDir)
        print(self.dataDir)

        self.online_dataset = []

    # ------------------------------------------------------
    def LoadDataForTraining(self, train_dataset=None):
        # trainset
        if train_dataset is None:
            train_dataset = np.load(self.dataDir + 'train_data/' + self.env_dim + '/state_0.npy').astype(np.float32)[
                600 * 2: 600 * 10, :]
        self.trainDataset = NNDataset(train_dataset.astype(np.float32))
        self.trainDataLoader = DataLoader(self.trainDataset, batch_size=64, shuffle=True, num_workers=4)
        # self.trainDataLoader = DataLoader(self.trainDataset, batch_size=train_dataset.shape[0], shuffle=True, num_workers=4)

    # ------------------------------------------------------
    def LoadDataForTest(self, test_dataset=None):
        # testset
        if test_dataset is None:
            test_dataset = np.load(self.dataDir + 'train_data/' + self.env_dim + '/state_0.npy').astype(np.float32)[
                600 * 0: 600 * 2, :]
        self.testDataset = NNDataset(test_dataset.astype(np.float32))
        self.testDataLoader = DataLoader(self.testDataset, batch_size=64, shuffle=False, num_workers=4)

    # ------------------------------------------------------
    def LoadModelWeights(self, file=None):
        if file is not None:
            if os.path.exists(self.nnWeightDir + "/" + file):
                print("Load Model path", self.nnWeightDir + "/" + file)
                self.model_J.load_state_dict(torch.load(self.nnWeightDir + "/" + file))
                print('Load previous model.', file)
            else:
                print('Warning: no model exists !')
        else:
            # offline_model = rospy.get_param("controller/offline_model")
            offline_model = args.controller_offline_model
            # if rospy.get_param("learning/is_test"):
            if args.learning_is_test:
                if os.path.exists(self.nnWeightDir + "/model_J.pth"):
                    self.model_J.load_state_dict(torch.load(self.nnWeightDir + "/model_J.pth"))
                    # print('Load previous model.')
                else:
                    print('Warning: no model exists !')
            else:
                if os.path.exists(self.nnWeightDir + offline_model + "/model_J.pth"):
                    self.model_J.load_state_dict(torch.load(self.nnWeightDir + offline_model + "/model_J.pth"))
                    # print('Load previous model.')
                else:
                    print('Warning: no model exists !')

            self.n_count = 0
            self.online_dataset = []

    # ------------------------------------------------------
    def SaveModelWeights(self):
        torch.save(self.model_J.state_dict(), self.nnWeightDir + "model_J_GNN_v8.pth")
        # print("Save models to ", self.nnWeightDir)

    # ------------------------------------------------------
    def Train(self, loadPreModel=False, n_epoch=50, save_model=True, rotation_augmentation=True):

        # if loadPreModel == False:  # use kmeans to calculate the initial value of mean and sigma of gaussian kernels
        #     if rotation_augmentation:
        #         self.model_J.GetMuAndBetaByKMeans(
        #                 self.relativeStateRepresentationTorch(dataRandomTransform(self.trainDataset.state_input)))
        #     else:
        #         self.model_J.GetMuAndBetaByKMeans(
        #                 self.relativeStateRepresentationTorch(self.trainDataset.state_input))
        # else:
        #     self.LoadModelWeights()

        log_file = "loss_GNN_v8_rot_coord_len_hid4.txt"
        with open(log_file, "w") as f:
            f.write("Epoch,Loss\n")

        # training
        loss_lst = []
        for epoch in range(0, n_epoch):
            accumLoss = 0.0
            numBatch = 0
            for batch_idx, (length, state_input, fps_vel, ends_vel) in enumerate(self.trainDataLoader):
                # rotation + length augmentation + length normalization
                # choose L0 = 1.0 (or your nominal rope length in meters)
                L0 = 1.0
                state_input, length_norm, fps_vel, ends_vel = dataRandomTransformWithLength(
                    state_input, length, fps_vel, ends_vel,
                    length_augment=True,   # set False if you only want rotation
                    L0=L0
                )

                 # ---- NEW: randomly hide contiguous and non-contiguous FP blocks in some samples ----
                state_input, fp_visible_mask = mask_feature_points_batch_combined_2(
                    state_input,
                    p_mask=0.25,
                    numFPs=args.DLO_num_FPs
                )

                # construct graph using *physical* length for radius
                # here we recover physical length: length_phys = length_norm * L0
                length_phys = length_norm * L0
                graph_lst = get_graph_batch(state_input, length_phys, fp_visible_mask, numFPs=args.DLO_num_FPs)
                graph_batch = Batch.from_data_list(graph_lst)

                graph_batch = graph_batch.to('cuda' if torch.cuda.is_available() else 'cpu')
                # print(graph_batch)

                # normalization
                ends_vel /= (torch.linalg.norm(fps_vel, dim=1).unsqueeze(1) + 1e-8)
                fps_vel /= (torch.linalg.norm(fps_vel, dim=1).unsqueeze(1) + 1e-8)  # avoid division by zero

                # data to GPU
                length = length.to('cuda' if torch.cuda.is_available() else 'cpu')
                state_input = state_input.to('cuda' if torch.cuda.is_available() else 'cpu')
                fps_vel = fps_vel.to('cuda' if torch.cuda.is_available() else 'cpu')
                ends_vel = ends_vel.to('cuda' if torch.cuda.is_available() else 'cpu')

                bmm_ends_vel = torch.reshape(ends_vel, (-1, 1, 12))
                bmm_fps_vel = torch.reshape(fps_vel, (-1, 1, self.numFPs * 3))

                # J_pred = self.model_J(state_input)
                J_pred = torch.reshape(
                    torch.reshape(self.model_J(graph_batch), (-1, args.DLO_num_FPs, 12, 3)).transpose(2, 3),
                    (-1, 3 * args.DLO_num_FPs, 12))
                # print(J_pred.shape)
                J_pred[:, :, [3, 4, 5, 9, 10, 11]] *= length.reshape(-1, 1, 1)  # N * T to get the final Jacobian
                # J_pred dim = [batch_size,30,12]
                J_pred_T = J_pred.transpose(1, 2)
                # J_pred_T = [batch_size,12,30]
                loss = self.criterion(bmm_fps_vel, torch.bmm(bmm_ends_vel, J_pred_T))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                accumLoss += loss.item()
                numBatch += 1

                # torch.save(J_pred, 'J_data.pt')
                # print("Tensor saved as 'tensor_data.pt'")
            print("epoch: ", epoch, " , Loss/train: ", accumLoss / numBatch)
            epoch_loss = accumLoss / numBatch
            loss_lst.append(epoch_loss)

            # Log loss immediately
            with open(log_file, "a") as f:
                f.write(f"{epoch + 1},{epoch_loss:.6f}\n")

        # save model
        if save_model:
            self.SaveModelWeights()

        # with open("loss_RBF.txt", "w") as f:
        #     for loss in loss_lst:
        #         f.write(f"{loss}\n")
        f.close()

    # ------------------------------------------------------
    def TestAndSaveResults(self):

        self.LoadModelWeights("model_J.pth")

        accumLoss = 0.0
        numBatch = 0
        for batch_idx, (length, state_input, fps_vel, ends_vel) in enumerate(self.testDataLoader):
            # state_input = self.relativeStateRepresentationTorch(state_input)

            # construct graph

            graph_lst = get_graph_batch(state_input, length, 1, numFPs=args.DLO_num_FPs)
            graph_batch = Batch.from_data_list(graph_lst)

            graph_batch = graph_batch.to('cuda' if torch.cuda.is_available() else 'cpu')

            length = length.to('cuda' if torch.cuda.is_available() else 'cpu')
            state_input = state_input.to('cuda' if torch.cuda.is_available() else 'cpu')
            fps_vel = fps_vel.to('cuda' if torch.cuda.is_available() else 'cpu')
            ends_vel = ends_vel.to('cuda' if torch.cuda.is_available() else 'cpu')

            bmm_ends_vel = torch.reshape(ends_vel, (-1, 1, 12))
            bmm_fps_vel = torch.reshape(fps_vel, (-1, 1, self.numFPs * 3))

            # J_pred = self.model_J(state_input)

            J_pred = torch.reshape(
                torch.reshape(self.model_J(graph_batch), (-1, args.DLO_num_FPs, 12, 3)).transpose(2, 3),
                (-1, 3 * args.DLO_num_FPs, 12))
            J_pred[:, :, [3, 4, 5, 9, 10, 11]] *= length.reshape(-1, 1, 1)

            J_pred_T = J_pred.transpose(1, 2)
            fps_vel_pred = torch.bmm(bmm_ends_vel, J_pred_T)

            testLoss = self.mse_criterion(fps_vel_pred, bmm_fps_vel)
            # testLoss = self.criterion(fps_vel_pred, bmm_fps_vel)

            accumLoss += testLoss.item()
            numBatch += 1

        # print("Loss/test: ", accumLoss/numBatch)
        print("Loss/test: ", accumLoss / len(self.testDataset))

        # test result 数据保存
        np.save(self.resultsDir + "nn_test/rbf/" + self.env_dim + "/dot_x_truth.npy",
                bmm_fps_vel.cpu().detach().numpy())
        np.save(self.resultsDir + "nn_test/rbf/" + self.env_dim + "/dot_x_pred.npy",
                fps_vel_pred.cpu().detach().numpy())

    # ------------------------------------------------------
    def OnlineLearningAndPredictJ(self, state, task_error=None):
        state = copy.copy(state)
        task_error = copy.copy(task_error)

        self.n_count += 1
        # parameters
        window_size = params_online_window_time * self.control_rate

        length = state[I.length_idx]
        state_input = state[I.state_input_idx].reshape(1, -1)  # one row matrix
        if task_error is None:
            task_error = np.zeros((self.numFPs * 3,), dtype='float32')

        # Because of the imperfection of the simulator, sometimes the DLO will wiggle to the other side very fast.
        # We don't want to include these outlier data in training, so we just discard the online data with too fast speed.
        fps_vel_norm = np.linalg.norm(state[I.fps_vel_idx])
        if fps_vel_norm > params_online_max_valid_fps_vel or fps_vel_norm < params_online_min_valid_fps_vel:
            # return the Jacobian without online updating
            length_torch = torch.tensor(length).to('cuda' if torch.cuda.is_available() else 'cpu')
            state_input_torch = self.relativeStateRepresentationTorch(torch.tensor(state_input)).to('cuda' if torch.cuda.is_available() else 'cpu')
            J_pred = self.model_J(state_input_torch)
            J_pred[:, :, [3, 4, 5, 9, 10, 11]] *= length_torch.reshape(-1, 1, 1)  # no normalize 要改 # RBF_abs 要改
            return J_pred.cpu().detach().numpy().reshape(3 * self.numFPs, 12)

        # normalize the velocities
        elif fps_vel_norm > params_online_fps_vel_thres:
            state[I.fps_vel_idx] /= (fps_vel_norm)
            state[I.ends_vel_idx] /= (fps_vel_norm)
            task_error *= (fps_vel_norm)
        else:
            state[I.fps_vel_idx] /= params_online_fps_vel_thres
            state[I.ends_vel_idx] /= params_online_fps_vel_thres
            task_error *= (params_online_fps_vel_thres)

        fps_vel = state[I.fps_vel_idx]
        ends_vel = state[I.ends_vel_idx]

        # --------------------------------------------------
        # update the NN weights
        # we use the SGD optimizer in PyTorch for online learning implementation to achieve faster computing speed.
        # Note that the following computing is mathematically equivalent to the online updating law in the paper.

        # learning rate: transform the learning rates  to the weights in the loss function
        weight_approx_e = np.sqrt(self.lr_approx_e / window_size / 2)
        if weight_approx_e == 0:
            weight_task_e = 0
        else:
            weight_task_e = self.lr_task_e / 2 / weight_approx_e

        # data preparation
        # latest step data
        length_torch = torch.tensor(length).to('cuda' if torch.cuda.is_available() else 'cpu')
        state_input_torch = self.relativeStateRepresentationTorch(torch.tensor(state_input)).to('cuda' if torch.cuda.is_available() else 'cpu')
        ends_vel_torch = torch.reshape(torch.tensor(ends_vel), (1, 1, 12)).to('cuda' if torch.cuda.is_available() else 'cpu')
        fps_vel_torch = torch.reshape(torch.tensor(fps_vel), (1, 1, 3 * self.numFPs)).to('cuda' if torch.cuda.is_available() else 'cpu')
        task_error_torch = torch.reshape(torch.tensor(task_error), (1, 1, 3 * self.numFPs)).to('cuda' if torch.cuda.is_available() else 'cpu')

        # previous data in sliding window
        if len(self.online_dataset) > 1:
            online_dataset = np.array(self.online_dataset)
            length_batch = online_dataset[:, I.length_idx]
            state_input_batch = online_dataset[:, I.state_input_idx]
            if len(state_input_batch.shape) == 1:
                state_input_batch = state_input_batch.reshape(1, -1)
            fps_vel_batch = online_dataset[:, I.fps_vel_idx]
            ends_vel_batch = online_dataset[:, I.ends_vel_idx]
            length_batch_torch = torch.tensor(length_batch).to('cuda' if torch.cuda.is_available() else 'cpu')
            state_input_batch_torch = self.relativeStateRepresentationTorch(torch.tensor(state_input_batch)).to('cuda' if torch.cuda.is_available() else 'cpu')
            ends_vel_batch_torch = torch.reshape(torch.tensor(ends_vel_batch), (-1, 1, 12)).to('cuda' if torch.cuda.is_available() else 'cpu')
            fps_vel_batch_torch = torch.reshape(torch.tensor(fps_vel_batch), (-1, 1, 3 * self.numFPs)).to('cuda' if torch.cuda.is_available() else 'cpu')

        # updating
        if (params_update_if_window_full is False) or (len(self.online_dataset) == window_size - 1):

            for epoch in range(int(self.online_update_rate / self.control_rate)):

                self.online_optimizer.zero_grad()

                # data at the current time
                J_pred = self.model_J(state_input_torch)
                J_pred[:, :, [3, 4, 5, 9, 10, 11]] *= length_torch.reshape(-1, 1, 1)
                J_pred_T = J_pred.transpose(1, 2)
                task_e = task_error_torch
                approx_e = fps_vel_torch - torch.bmm(ends_vel_torch, J_pred_T)
                loss = self.mse_criterion(weight_approx_e * approx_e + weight_task_e * task_e,
                                          torch.zeros(approx_e.shape).to('cuda' if torch.cuda.is_available() else 'cpu'))

                loss.backward()

                # previous data stored in the sliding window
                if len(self.online_dataset) > 1:
                    J_pred = self.model_J(state_input_batch_torch)
                    J_pred[:, :, [3, 4, 5, 9, 10, 11]] *= length_batch_torch.reshape(-1, 1, 1)
                    J_pred_T = J_pred.transpose(1, 2)
                    loss = self.lr_approx_e / window_size / 2 * self.mse_criterion(fps_vel_batch_torch,
                                                                                   torch.bmm(ends_vel_batch_torch,
                                                                                             J_pred_T))

                    loss.backward()

                # do the update
                self.online_optimizer.step()
        # --------------------------------------------------

        # store the data at the current time in the sliding window
        self.online_dataset.append(state)
        # remove the earliest data in the sliding window
        if len(self.online_dataset) > window_size - 1:
            self.online_dataset.pop(0)

        # return the updated Jacobian matrix
        J_pred = self.model_J(state_input_torch)
        J_pred[:, :, [3, 4, 5, 9, 10, 11]] *= length_torch.reshape(-1, 1, 1)

        return J_pred.cpu().detach().numpy().reshape(3 * self.numFPs, 12)

    # --------------------------------------------------------------------
    def calcNextEndsPose(self, current_ends_pose, ends_vel, delta_t=0.1):
        left_end_pos = current_ends_pose[:, 0:3]
        left_end_quat = current_ends_pose[:, 3:7]
        right_end_pos = current_ends_pose[:, 7:10]
        right_end_quat = current_ends_pose[:, 10:14]

        left_end_lvel = ends_vel[:, 0:3]
        left_end_avel = ends_vel[:, 3:6]
        right_end_lvel = ends_vel[:, 6:9]
        right_end_avel = ends_vel[:, 9:12]

        next_left_end_pos = left_end_pos + left_end_lvel * delta_t
        next_right_end_pos = right_end_pos + right_end_lvel * delta_t

        left_end_ori = sciR.from_quat(left_end_quat)
        left_end_delta_ori = sciR.from_rotvec(left_end_avel * delta_t)
        next_left_end_ori = left_end_delta_ori * left_end_ori
        next_left_end_quat = next_left_end_ori.as_quat()

        right_end_ori = sciR.from_quat(right_end_quat)
        right_end_delta_ori = sciR.from_rotvec(right_end_avel * delta_t)
        next_right_end_ori = right_end_delta_ori * right_end_ori
        next_right_end_quat = next_right_end_ori.as_quat()

        next_ends_pose = np.concatenate(
            [next_left_end_pos, next_left_end_quat, next_right_end_pos, next_right_end_quat], axis=1)
        return next_ends_pose

    # ------------------------------------------------------
    def predNextFPsPositions(self, length, fps_pos, ends_pose, ends_vel, delta_t):
        if fps_pos.ndim == 1 or ends_pose.ndim == 1 or ends_vel.ndim == 1:
            length = length.reshape(1, -1)
            fps_pos = fps_pos.reshape(1, -1)
            ends_pose = ends_pose.reshape(1, -1)
            ends_vel = ends_vel.reshape(1, -1)

        fps_vel_pred = self.predFPsVelocities(length, fps_pos, ends_pose, ends_vel)
        next_fps_pos = fps_pos + delta_t * fps_vel_pred
        return next_fps_pos

    # ------------------------------------------------------
    def predFPsVelocities(self, length, fps_pos, ends_pose, ends_vel):
        state_input = np.concatenate([fps_pos, ends_pose], axis=1).astype('float32')  # one row matrix
        # np array to torch tensor
        state_input_torch = self.relativeStateRepresentationTorch(torch.tensor(state_input)).to('cuda' if torch.cuda.is_available() else 'cpu')
        ends_vel_torch = torch.reshape(torch.tensor(ends_vel.astype('float32')), (-1, 1, 12)).to('cuda' if torch.cuda.is_available() else 'cpu')
        length_torch = torch.tensor(length.astype('float32')).to('cuda' if torch.cuda.is_available() else 'cpu')

        # predict the feature velocities
        J_pred = self.model_J(state_input_torch)
        J_pred[:, :, [3, 4, 5, 9, 10, 11]] *= length_torch.reshape(-1, 1, 1)

        J_pred_T = J_pred.transpose(1, 2)
        fps_vel_pred = torch.bmm(ends_vel_torch, J_pred_T).cpu().detach().numpy().reshape(-1, self.numFPs * 3)

        return fps_vel_pred

    # ------------------------------------------------------
    # state representation preprocess
    def relativeStateRepresentationTorch(self, state_input):
        b_numpy = False
        if type(state_input) is np.ndarray:
            state_input = torch.tensor(state_input)
            b_numpy = True

        numFPs = self.numFPs
        left_end_pos = state_input[:, 3 * numFPs: 3 * numFPs + 3]
        left_end_quat = state_input[:, 3 * numFPs + 3: 3 * numFPs + 7]
        right_end_pos = state_input[:, 3 * numFPs + 7: 3 * numFPs + 10]
        right_end_quat = state_input[:, 3 * numFPs + 10: 3 * numFPs + 14]

        fps_pos = state_input[:, 0: 3 * numFPs].reshape(-1, numFPs, 3)
        if len(fps_pos.shape) == 1:  # reshape fps_pos from vector to one-row matrix
            fps_pos = fps_pos.unsqueeze(0)

        fps_pos_r = torch.zeros(fps_pos.shape)
        fps_pos_r[:, 1:, :] = (fps_pos[:, 1:, :] - fps_pos[:, 0:-1, :])
        fps_pos_r[:, 1:, :] /= torch.linalg.norm(fps_pos_r[:, 1:, :], dim=2).unsqueeze(2)
        fps_pos_r = fps_pos_r.reshape(-1, 3 * numFPs)

        right_end_pos_r = (right_end_pos - left_end_pos)
        right_end_pos_r /= torch.linalg.norm(right_end_pos_r, dim=1).unsqueeze(1)

        relative_state_input = torch.cat((fps_pos_r, right_end_pos_r, left_end_quat, right_end_quat),
                                         dim=1)  # [60000,41], reduce input vector dimension by 3, combine q1,q2
        if b_numpy:
            return relative_state_input.numpy()
        else:
            return relative_state_input


# --------------------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # project_dir = rospy.get_param("project_dir")
    # env_dim = rospy.get_param("env/dimension")

    project_dir = args.project_dir
    env_dim = args.env_dimension

    # training dataset (from 10 DLOs)
    train_dataset = np.empty((0, I.state_dim)).astype("float32")
    for j in range(1, 11):
        # state [6000, 117]
        # state = np.load(project_dir + "data/train_data/"+ env_dim + "/state_" + str(j) + ".npy").astype("float32")[: 6000, :]
        state = np.load(os.path.join(project_dir, "data", "train_data", env_dim, f"state_{j}.npy")).astype("float32")[
            :6000, :]

        train_dataset = np.concatenate([train_dataset, state], axis=0)

    # train_dataset [60000,117]

    trainer = JacobianPredictor()
    trainer.LoadDataForTraining(train_dataset)

    trainer.Train(loadPreModel=False, n_epoch=50, rotation_augmentation=False)

# # test data, test for DLO 0
#  test_dataset = np.empty((0, I.state_dim)).astype("float32")
#  test_state = np.load(os.path.join(project_dir, "data", "train_data", env_dim, "state_0.npy")).astype("float32")[:, :]
#  test_dataset = np.concatenate([test_dataset, test_state], axis=0)
#  print(test_dataset.shape)
#  tester = JacobianPredictor()
#  tester.LoadDataForTest(test_dataset)
#  tester.TestAndSaveResults()


