# #!/home/xinge/miniconda3/envs/dlo/bin/python
# # /usr/bin/env python
# import numpy as np
# from matplotlib import pyplot as plt
# import os
# import time
# from sklearn.cluster import KMeans
# import copy
# # import rospy
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from scipy.spatial.transform import Rotation as sciR

# # import torch_rbf as rbf # reference: https://github.com/JeremyLinux/PyTorch-Radial-Basis-Function-Layer


# from utils.data_augmentation import dataRandomTransform
# from utils.state_index import I

# import json

# from torch_geometric.data import Data, Batch
# import torch_geometric as pyg
# from torch_geometric.nn import radius_graph
# from GN_model_DLO import MySubEdge, MySubNode
# from torch_geometric.nn import global_mean_pool

# import argparse
# parser = argparse.ArgumentParser(description="DLO Parser")

# def load_args_from_file(filepath='/home/jessica/shape_control_DLO_2/ws_dlo/src/dlo_system_pkg/config/config.json'):
#     with open(filepath, 'r') as file:
#         args_dict = json.load(file)
#     # Convert dictionary back to Namespace
#     args = argparse.Namespace(**args_dict)
#     return args

# args = load_args_from_file()
# # print(args)

# params_online_window_time = 2  # unit: second
# params_online_max_valid_fps_vel = 0.3
# params_online_fps_vel_thres = 0.01
# params_online_min_valid_fps_vel = 0.00
# params_update_if_window_full = False
# tag = 'trial'


# class Args:
#     def __init__(self, **kwargs):
#         self.__dict__.update(kwargs)

# args_GN = Args(
#     lr=1e-4
# )


# # ----------------------------------------------------------------------------------------------------------
# # ----------------------------------------------------------------------------------------------------------
# # ----------------------------------------------------------------------------------------------------------
# #seperate the x, s, v, x_dot from the state vector (dim=117)
# class NNDataset(Dataset):
#     def __init__(self, state):
#         self.data_num = state.shape[0]
#         self.length = state[:, I.length_idx]
#         self.state_input = state[:, I.state_input_idx]
#         self.fps_vel = state[:, I.fps_vel_idx]
#         self.ends_vel = state[:, I.ends_vel_idx]
    
#     def __getitem__(self, index):
#         return self.length[index], self.state_input[index], self.fps_vel[index], self.ends_vel[index]

#     def __len__(self):
#         return self.data_num


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GINEConv, global_mean_pool

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from torch_scatter import scatter_add, scatter_mean
# from torch_geometric.utils import to_dense_batch
# from utils.utils_graph import get_graph_batch_RBF_comparable_fast

# class MP1_NodeHead_JacobianNet(nn.Module):
#     """
#     1 message passing layer (manual) + node-specific heads for Ji.

#     Expects PyG Batch `data` with:
#       data.x         : [total_nodes, node_in]
#       data.edge_index: [2, total_edges]
#       data.edge_attr : [total_edges, edge_in]
#       data.batch     : [total_nodes]   (graph id per node)
#       data.u         : [num_graphs, u_dim]  (global, e.g., 11)

#     Assumption:
#       - Each graph has exactly numFPs nodes (FPS only), no end nodes.
#     Output:
#       J: [B, 3*numFPs, 12]
#     """
#     def __init__(
#         self,
#         numFPs: int,
#         node_in: int,
#         edge_in: int,
#         u_dim: int = 11,
#         hidden: int = 256,
#         agg: str = "mean",   # "mean" or "sum"
#     ):
#         super().__init__()
#         self.numFPs = numFPs
#         self.node_in = node_in
#         self.edge_in = edge_in
#         self.u_dim = u_dim
#         self.hidden = hidden
#         self.agg = agg

#         # --- A) edge MLP: edge_attr -> hidden (one layer) ---
#         # self.edge_mlp = nn.Linear(edge_in, hidden, bias=False)

#         self.edge_mlp  = nn.Sequential(
#             nn.Linear(edge_in, hidden, bias=False),
#             nn.ReLU(inplace=True)
#             # nn.Linear(hidden, hidden,bias=False),
#             # nn.ReLU(inplace=True),
#         )

#         # --- B) node MLP: x -> hidden (one layer) ---
#         # self.node_mlp = nn.Linear(node_in, hidden, bias=False)

#         self.node_mlp = nn.Sequential(
#             nn.Linear(node_in, hidden, bias=False),
#             nn.ReLU(inplace=True)
#             # nn.Linear(hidden, hidden,bias=False),
#             # nn.ReLU(inplace=True),
#             )

#         # --- C) fusion MLP: [node_h, msg_h, u] -> hidden ---
#         self.fuse = nn.Sequential(
#             nn.Linear(hidden+hidden, hidden, bias=False),
#             nn.ReLU(inplace=True),
#             # nn.Linear(hidden, hidden, bias=False),
#             # nn.ReLU(inplace=True),
#         )

#         # # --- D) node-specific heads: hidden -> 36 (= 12*3) per node index i ---
#         # self.heads = nn.ModuleList([nn.Linear(hidden, 36, bias=False) for _ in range(numFPs)])

#         # --- D) shared head: hidden -> 36 (=12*3) for every node ---
#         self.head = nn.Linear(hidden, 36, bias=False)

#     def forward(self, data):
#         if not hasattr(data, "u"):
#             raise ValueError("data.u is required (global features), but missing.")
#         if data.x.dim() != 2:
#             raise ValueError(f"data.x must be [N, node_in], got {data.x.shape}")

#         x = data.x                      # [Nt, node_in]
#         edge_index = data.edge_index    # [2, Et]
#         eattr = data.edge_attr          # [Et, edge_in]
#         batch = data.batch              # [Nt]
#         u = data.u                      # [B, u_dim]

#         Nt = x.size(0)
#         B = u.size(0)

#         # 1) node embedding
#         # node_h = F.relu(self.node_mlp(x), inplace=True)  # [Nt, hidden]

#         node_h = self.node_mlp(x)
#         # node_h = x

#         # 2) edge message from edge_attr only (as you requested)
#         # msg_e = F.relu(self.edge_mlp(eattr), inplace=True)  # [Et, hidden]

#         msg_e = self.edge_mlp(eattr)

#         # 3) aggregate messages to destination nodes
#         src, dst = edge_index[0], edge_index[1]  # [Et], [Et]
#         if self.agg == "sum":
#             msg_h = scatter_add(msg_e, dst, dim=0, dim_size=Nt)   # [Nt, hidden]
#         elif self.agg == "mean":
#             msg_h = scatter_mean(msg_e, dst, dim=0, dim_size=Nt)  # [Nt, hidden]
#         else:
#             raise ValueError(f"Unknown agg='{self.agg}', use 'mean' or 'sum'.")

#         # # ---------------- GLOBAL PART (NO NEW LAYERS) ----------------
#         # # 4) global node = pooled node_h (or pooled msg_h, either is fine)
#         # if self.agg == "mean":
#         #     g = scatter_mean(node_h, batch, dim=0, dim_size=B)  # [B, H]
#         # else:
#         #     g = scatter_add(node_h, batch, dim=0, dim_size=B)
#         #
#         # g_node = g[batch]  # [Nt, H]
#         # u_node = u[batch]  # [Nt, u_dim]


#         # 5) fuse
#         z = torch.cat([node_h, msg_h], dim=-1)        # [Nt, 2H+u_dim]
#         # z = torch.cat([node_h], dim=-1)  # [Nt, 2H+u_dim]
#         h = self.fuse(z)                                      # [Nt, hidden]
#         # h = z

#         # # 6) reshape to [B, N, hidden] to apply node-specific heads
#         # # safest: use to_dense_batch (works even if something weird happens)
#         # h_dense, mask = to_dense_batch(h, batch=batch, max_num_nodes=self.numFPs)  # [B, N, hidden]
#         # if h_dense.size(1) != self.numFPs:
#         #     raise ValueError(f"Expected numFPs={self.numFPs} nodes per graph, got {h_dense.size(1)}")
#         #
#         # # optional: ensure no missing nodes
#         # if not mask.all():
#         #     # If you truly have missing nodes, you need a policy (zero pad is okay,
#         #     # but node-specific heads become ambiguous). For now, we fail loudly.
#         #     raise ValueError("Found missing nodes in a graph (mask not all True).")
#         #
#         # # 7) node-specific heads
#         # per_node_out = []
#         # for i in range(self.numFPs):
#         #     out_i = self.heads[i](h_dense[:, i, :])  # [B, 36]
#         #     per_node_out.append(out_i)
#         #
#         # out = torch.stack(per_node_out, dim=1)  # [B, N, 36]
#         #
#         # # 8) convert to J: [B, 3N, 12]
#         # out = out.view(B, self.numFPs, 12, 3).transpose(2, 3)  # [B, N, 3, 12]
#         # J = out.reshape(B, 3 * self.numFPs, 12)                # [B, 3N, 12]
#         # return J

#         # 6) reshape to [B, N, hidden]
#         h_dense, mask = to_dense_batch(h, batch=batch, max_num_nodes=self.numFPs)  # [B,N,H]
#         if h_dense.size(1) != self.numFPs:
#             raise ValueError(f"Expected numFPs={self.numFPs} nodes per graph, got {h_dense.size(1)}")

#         if not mask.all():
#             raise ValueError("Found missing nodes in a graph (mask not all True).")

#         # 7) shared head for all nodes (vectorized)
#         out = self.head(h_dense)                     # [B, N, 36]

#         # 8) convert to J: [B, 3N, 12]
#         out = out.view(B, self.numFPs, 12, 3).transpose(2, 3)  # [B,N,3,12]
#         J = out.reshape(B, 3 * self.numFPs, 12)                # [B,3N,12]
#         return J



# numFPs = args.DLO_num_FPs
# # ----------------------------------------------------------------------------------------------------------
# # ----------------------------------------------------------------------------------------------------------
# class JacobianPredictor(object):
#     #
#     # numFPs = rospy.get_param("DLO/num_FPs")
#     # projectDir = rospy.get_param("project_dir")
#     # online_learning_rate = rospy.get_param("controller/online_learning/learning_rate")
#     # lr_task_e = online_learning_rate
#     # lr_approx_e = online_learning_rate * rospy.get_param("controller/online_learning/weight_ratio")
#     # env = rospy.get_param("env/sim_or_real")
#     # env_dim = rospy.get_param("env/dimension")
#     # control_rate = rospy.get_param("ros_rate/env_rate")
#     # online_update_rate = rospy.get_param("ros_rate/online_update_rate")

#     numFPs = args.DLO_num_FPs
#     projectDir = args.project_dir
#     online_learning_rate = args.controller_online_learning_learning_rate
#     lr_task_e = online_learning_rate
#     lr_approx_e = online_learning_rate * args.controller_online_learning_weight_ratio
#     env = args.env_sim_or_real
#     env_dim = args.env_dimension
#     control_rate = args.ros_rate_env_rate
#     online_update_rate = args.ros_rate_online_update_rate
    
#     # ------------------------------------------------------
#     def __init__(self, num_hidden_unit=256):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        
#         self.bTrainMuBeta = True

#         # self.model_J = MySimulator().to(self.device)

#         self.model_J = MP1_NodeHead_JacobianNet(
#             numFPs=10,
#             node_in=19,
#             edge_in=4+19+19,
#             u_dim=11,
#             hidden=256,
#             agg='mean',
#         ).to(self.device)

#         print("Model J")
#         print(self.model_J)

#         self.optimizer = torch.optim.Adam(self.model_J.parameters(), lr=1e-3)

#         # for offline learning
#         # torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99, last_epoch=-1, verbose=False)
#         self.criterion = torch.nn.SmoothL1Loss(reduction='mean', beta=0.001)
#         # for online learning
#         # self.online_optimizer = torch.optim.SGD([{'params': self.model_J.fc2.parameters()}], lr=1.0/self.online_update_rate)
#         self.online_optimizer = torch.optim.SGD(self.model_J.head.parameters(), lr=1.0 / self.online_update_rate)
#         self.mse_criterion = torch.nn.MSELoss(reduction='sum')

#         # if rospy.get_param("learning/is_test"):
#         if args.learning_is_test:
#             # self.nnWeightDir = self.projectDir + 'ws_dlo/src/dlo_manipulation_pkg/models_test/rbfWeights/' + self.env_dim + '/'
#             self.nnWeightDir = os.path.join(
#                 self.projectDir,
#                 'ws_dlo', 'src', 'dlo_manipulation_pkg', 'models_test',
#                 'gnnWeights', self.env_dim, ''
#             )
#         else:
#             # self.nnWeightDir = self.projectDir + 'ws_dlo/src/dlo_manipulation_pkg/models/rbfWeights/' + self.env_dim + '/'
#             self.nnWeightDir = os.path.join(
#                 self.projectDir,
#                 'ws_dlo', 'src', 'dlo_manipulation_pkg', 'models',
#                 'gnnWeights', self.env_dim, ''
#             )
#         # self.resultsDir = self.projectDir + 'results/' + self.env + '/'
#         self.resultsDir = os.path.join(self.projectDir, 'results', self.env , '')
#         # self.dataDir = self.projectDir +'data/'
#         self.dataDir = os.path.join(self.projectDir , 'data', '')

#         print('Directory')
#         print(self.nnWeightDir)
#         print(self.resultsDir)
#         print(self.dataDir)

#         self.online_dataset = []

        
#     # ------------------------------------------------------
#     def LoadDataForTraining(self, train_dataset=None):
#         # trainset
#         if train_dataset is None:
#             train_dataset = np.load(self.dataDir + 'train_data/' + self.env_dim + '/state_0.npy').astype(np.float32)[600*2 : 600*10, :]
#         self.trainDataset = NNDataset(train_dataset.astype(np.float32))
#         self.trainDataLoader = DataLoader(self.trainDataset, batch_size=512, shuffle=True, num_workers=0)
#         # self.trainDataLoader = DataLoader(self.trainDataset, batch_size=train_dataset.shape[0], shuffle=True, num_workers=0)

#     # ------------------------------------------------------
#     def LoadDataForTest(self, test_dataset=None):
#         # testset
#         if test_dataset is None:
#             test_dataset = np.load(self.dataDir + 'train_data/' + self.env_dim + '/state_0.npy').astype(np.float32)[600*0 : 600*2, :]
#         self.testDataset = NNDataset(test_dataset.astype(np.float32))
#         self.testDataLoader = DataLoader(self.testDataset, batch_size=512, shuffle=False, num_workers=0)


#     # ------------------------------------------------------
#     def LoadModelWeights(self, file=None):
#         offline_model = args.controller_offline_model  # 10*6 10*2 ...
#         if file is not None:
#             if os.path.exists(self.nnWeightDir + offline_model + "/" + file):
#                 print("Load Model path", self.nnWeightDir + offline_model + "/" + file)
#                 ckpt_path = os.path.join(self.nnWeightDir, offline_model, file)
#                 self.model_J.load_state_dict(torch.load(ckpt_path, map_location=self.device))

#                 print('Load previous model.', file)
#             else:
#                 print('Warning: no model exists !')
#         else:
#             # offline_model = rospy.get_param("controller/offline_model")
#             offline_model = args.controller_offline_model
#             # if rospy.get_param("learning/is_test"):
#             if args.learning_is_test:
#                 if os.path.exists(self.nnWeightDir  + "/model_J.pth"):
#                     ckpt_path = os.path.join(self.nnWeightDir, "model_J.pth")
#                     self.model_J.load_state_dict(torch.load(ckpt_path, map_location=self.device))

#                     # print('Load previous model.')
#                 else:
#                     print('Warning: no model exists !')
#             else:
#                 if os.path.exists(self.nnWeightDir + offline_model + "/model_J.pth"):
#                     ckpt_path = os.path.join(self.nnWeightDir, offline_model, "model_J.pth")
#                     self.model_J.load_state_dict(torch.load(ckpt_path, map_location=self.device))

#                     # print('Load previous model.')
#                 else:
#                     print('Warning: no model exists !')

#             self.n_count = 0
#             self.online_dataset = []

    
#     # ------------------------------------------------------
#     def SaveModelWeights(self, path=""):
#         os.makedirs(self.nnWeightDir, exist_ok=True)
#         if path:
#             torch.save(self.model_J.state_dict(), path)
#         else:
#             torch.save(self.model_J.state_dict(),
#                        os.path.join(self.nnWeightDir, args.controller_offline_model, "model_J_GNN_NN.pth"))
#         # print("Save models to ", self.nnWeightDir)


#     # ------------------------------------------------------
#     def Train(self, loadPreModel=False, n_epoch=50, save_model=True, rotation_augmentation=True):

#         # if loadPreModel == False:  # use kmeans to calculate the initial value of mean and sigma of gaussian kernels
#         #     if rotation_augmentation:
#         #         self.model_J.GetMuAndBetaByKMeans(
#         #                 self.relativeStateRepresentationTorch(dataRandomTransform(self.trainDataset.state_input)))
#         #     else:
#         #         self.model_J.GetMuAndBetaByKMeans(
#         #                 self.relativeStateRepresentationTorch(self.trainDataset.state_input))
#         # else:
#         #     self.LoadModelWeights()

#         log_file = f"loss_GNN_NN_{tag}.txt"
#         with open(log_file, "w") as f:
#             f.write("Epoch,Loss\n")

#         # training
#         loss_lst = []
#         for epoch in range(0, n_epoch):
#             accumLoss = 0.0
#             numBatch = 0

#             edge_cache = {}  # keep outside loop
#             for batch_idx, (length, state_input, fps_vel, ends_vel) in enumerate(self.trainDataLoader):       
#                 # data augmentation
#                 if rotation_augmentation:
#                     state_input, fps_vel, ends_vel = dataRandomTransform(state_input, fps_vel, ends_vel)
#                 # state_input = self.relativeStateRepresentationTorch(state_input)

#                 # construct graph

#                 # graph_lst = get_graph_batch(state_input, length, numFPs=args.DLO_num_FPs)
#                 # graph_batch = Batch.from_data_list(graph_lst)
#                 #
#                 # graph_batch = graph_batch.to(self.device)


#                 state_input = state_input.to(self.device)
#                 length = length.to(self.device)

#                 data, edge_cache = get_graph_batch_RBF_comparable_fast(
#                     state_input, length, numFPs=self.numFPs,
#                     edge_index_cache=edge_cache,
#                 )
#                 # data.x: [B*N, ...], data.edge_index: [2, B*E], data.batch: [B*N], data.u: [B,11]

#                 # print(graph_batch)

#                 # normalization
#                 ends_vel /= (torch.linalg.norm(fps_vel, dim=1).unsqueeze(1) + 1e-8)
#                 fps_vel /= (torch.linalg.norm(fps_vel, dim=1).unsqueeze(1) + 1e-8) # avoid division by zero

#                 # data to GPU
#                 length = length.to(self.device)
#                 state_input = state_input.to(self.device)
#                 fps_vel = fps_vel.to(self.device)
#                 ends_vel = ends_vel.to(self.device)

#                 bmm_ends_vel = torch.reshape(ends_vel, (-1, 1, 12))
#                 bmm_fps_vel = torch.reshape(fps_vel, (-1, 1, self.numFPs * 3))


#                 J_pred = self.model_J(data)                              # [B,3N,12]
#                 J_pred[:, :, [3, 4, 5, 9, 10, 11]] *= length.view(-1, 1, 1)

#                 J_pred_T = J_pred.transpose(1, 2)                          # [B,12,3N]
#                 loss = self.criterion(bmm_fps_vel, torch.bmm(bmm_ends_vel, J_pred_T))


#                 # # J_pred = self.model_J(state_input)
#                 # J_pred = torch.reshape(torch.reshape(self.model_J(graph_batch), (-1, args.DLO_num_FPs, 12, 3)).transpose(2, 3), (-1, 3 * args.DLO_num_FPs, 12))
#                 #
#                 # # print(J_pred.shape)
#                 # J_pred[:, :, [3, 4, 5, 9, 10, 11]] *= length.reshape(-1, 1, 1)  # N * T to get the final Jacobian
#                 # # J_pred dim = [batch_size,30,12]
#                 # J_pred_T = J_pred.transpose(1, 2)
#                 # # J_pred_T = [batch_size,12,30]
#                 # loss = self.criterion(bmm_fps_vel, torch.bmm(bmm_ends_vel, J_pred_T))

#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()
#                 accumLoss += loss.item()
#                 numBatch += 1

#                 # torch.save(J_pred, 'J_data.pt')
#                 # print("Tensor saved as 'tensor_data.pt'")
#             print("epoch: ", epoch, " , Loss/train: ", accumLoss/numBatch)
#             epoch_loss = accumLoss/numBatch
#             loss_lst.append(epoch_loss)

#             # Log loss immediately
#             with open(log_file, "a") as f:
#                 f.write(f"{epoch + 1},{epoch_loss:.6f}\n")

#             if save_model and epoch%20==0:
#                 self.SaveModelWeights(os.path.join(self.nnWeightDir, args.controller_offline_model, f"model_J_GNN_{tag}_{epoch}.pth"))

#         # save final model
#         if save_model:
#             self.SaveModelWeights()

#         # with open("loss_RBF.txt", "w") as f:
#         #     for loss in loss_lst:
#         #         f.write(f"{loss}\n")
#         f.close()


#     # ------------------------------------------------------
#     def TestAndSaveResults(self):

#         self.LoadModelWeights(f"model_J_GNN_NN.pth")

#         accumLoss = 0.0
#         numBatch = 0
#         edge_cache = {}  # keep outside loop
#         for batch_idx, (length, state_input, fps_vel, ends_vel) in enumerate(self.testDataLoader):
#             # state_input = self.relativeStateRepresentationTorch(state_input)


#             # construct graph

#             # graph_lst = get_graph_batch(state_input, length, numFPs=args.DLO_num_FPs)
#             # graph_batch = Batch.from_data_list(graph_lst)
#             #
#             # graph_batch = graph_batch.to(self.device)

#             state_input = state_input.to(self.device)
#             length = length.to(self.device)

#             data, edge_cache = get_graph_batch_RBF_comparable_fast(
#                 state_input, length, numFPs=self.numFPs,
#                 edge_index_cache=edge_cache,
#             )


#             length = length.to(self.device)
#             state_input = state_input.to(self.device)
#             fps_vel = fps_vel.to(self.device)
#             ends_vel = ends_vel.to(self.device)


#             bmm_ends_vel = torch.reshape(ends_vel, (-1, 1, 12))
#             bmm_fps_vel = torch.reshape(fps_vel, (-1, 1, self.numFPs * 3))

#             # J_pred = self.model_J(state_input)

#             # J_pred = torch.reshape(torch.reshape(self.model_J(graph_batch), (-1, args.DLO_num_FPs, 12, 3)).transpose(2, 3),
#             #                        (-1, 3 * args.DLO_num_FPs, 12))

#             J_pred = self.model_J(data)  # [B,3N,12]

#             J_pred[:, :, [3, 4, 5, 9, 10, 11]] *= length.reshape(-1, 1, 1)

#             J_pred_T = J_pred.transpose(1, 2)
#             fps_vel_pred = torch.bmm(bmm_ends_vel, J_pred_T)

#             # testLoss = self.mse_criterion(fps_vel_pred, bmm_fps_vel)
#             # testLoss = self.criterion(fps_vel_pred, bmm_fps_vel)

#             eps = 1e-8  # avoid division by zero

#             error = fps_vel_pred - bmm_fps_vel

#             rel_l2_error = (
#                     torch.norm(error, p=2, dim=-1) /
#                     (torch.norm(bmm_fps_vel, p=2, dim=-1)+eps)
#             )

#             # average over batch / time / nodes
#             testLoss = rel_l2_error.sum() * 100.0

#             accumLoss += testLoss.item()
#             numBatch += 1

#         # print("Loss/test: ", accumLoss/numBatch)
#         print("Loss/test: ", accumLoss/len(self.testDataset))

#         # test result 数据保存
#         np.save(self.resultsDir + "nn_test/rbf/" + self.env_dim + "/dot_x_truth.npy", bmm_fps_vel.cpu().detach().numpy()) 
#         np.save(self.resultsDir + "nn_test/rbf/" + self.env_dim + "/dot_x_pred.npy", fps_vel_pred.cpu().detach().numpy())

#     # ------------------------------------------------------
#     def OnlineLearningAndPredictJ(self, state, task_error=None):
#         state = copy.copy(state)
#         task_error = copy.copy(task_error)
        
#         self.n_count  += 1
#         # parameters
#         window_size = params_online_window_time * self.control_rate

#         length = state[I.length_idx]
#         state_input = state[I.state_input_idx].reshape(1, -1) # one row matrix
#         if task_error is None:
#             task_error = np.zeros((self.numFPs * 3, ), dtype='float32')

#         # Because of the imperfection of the simulator, sometimes the DLO will wiggle to the other side very fast.
#         # We don't want to include these outlier data in training, so we just discard the online data with too fast speed.
#         fps_vel_norm = np.linalg.norm(state[I.fps_vel_idx])
#         if fps_vel_norm > params_online_max_valid_fps_vel or  fps_vel_norm < params_online_min_valid_fps_vel:
#             # return the Jacobian without online updating
#             length_torch = torch.tensor(length).to(self.device)
#             state_input_torch = self.relativeStateRepresentationTorch(torch.tensor(state_input)).to(self.device)
#             J_pred = self.model_J(state_input_torch)
#             J_pred[:, :, [3, 4, 5, 9, 10, 11]] *= length_torch.reshape(-1, 1, 1)  # no normalize 要改 # RBF_abs 要改
#             return J_pred.cpu().detach().numpy().reshape(3 * self.numFPs, 12)

#         # normalize the velocities
#         elif fps_vel_norm > params_online_fps_vel_thres:
#             state[I.fps_vel_idx]  /= (fps_vel_norm)
#             state[I.ends_vel_idx]  /= (fps_vel_norm)
#             task_error *= (fps_vel_norm)
#         else:
#             state[I.fps_vel_idx] /= params_online_fps_vel_thres
#             state[I.ends_vel_idx]  /= params_online_fps_vel_thres
#             task_error *= (params_online_fps_vel_thres) 

#         fps_vel = state[I.fps_vel_idx]
#         ends_vel = state[I.ends_vel_idx]
            
#         # --------------------------------------------------
#         # update the NN weights
#         # we use the SGD optimizer in PyTorch for online learning implementation to achieve faster computing speed. 
#         # Note that the following computing is mathematically equivalent to the online updating law in the paper.

#         # learning rate: transform the learning rates  to the weights in the loss function
#         weight_approx_e = np.sqrt(self.lr_approx_e /  window_size / 2)
#         if weight_approx_e == 0:
#             weight_task_e = 0
#         else:
#             weight_task_e = self.lr_task_e / 2 /  weight_approx_e

#         # data preparation
#         # latest step data
#         length_torch = torch.tensor(length).to(self.device)
#         state_input_torch = self.relativeStateRepresentationTorch(torch.tensor(state_input)).to(self.device)
#         ends_vel_torch = torch.reshape(torch.tensor(ends_vel), (1, 1, 12)).to(self.device)
#         fps_vel_torch = torch.reshape(torch.tensor(fps_vel), (1, 1, 3 * self.numFPs)).to(self.device)
#         task_error_torch = torch.reshape(torch.tensor(task_error), (1, 1, 3 * self.numFPs)).to(self.device)

#         # previous data in sliding window
#         if len(self.online_dataset) > 1:
#             online_dataset = np.array(self.online_dataset)
#             length_batch = online_dataset[:, I.length_idx]
#             state_input_batch = online_dataset[:, I.state_input_idx]
#             if len(state_input_batch.shape) == 1:
#                 state_input_batch = state_input_batch.reshape(1, -1)
#             fps_vel_batch = online_dataset[:, I.fps_vel_idx]
#             ends_vel_batch = online_dataset[:, I.ends_vel_idx]
#             length_batch_torch = torch.tensor(length_batch).to(self.device)
#             state_input_batch_torch = self.relativeStateRepresentationTorch(torch.tensor(state_input_batch)).to(self.device)
#             ends_vel_batch_torch = torch.reshape(torch.tensor(ends_vel_batch), (-1, 1, 12)).to(self.device)
#             fps_vel_batch_torch = torch.reshape(torch.tensor(fps_vel_batch), (-1, 1, 3 * self.numFPs)).to(self.device)

#         # updating
#         if (params_update_if_window_full is False) or (len(self.online_dataset) == window_size - 1):

#             for epoch in range(int(self.online_update_rate / self.control_rate)):

#                 self.online_optimizer.zero_grad()

#                 # data at the current time
#                 J_pred = self.model_J(state_input_torch)
#                 J_pred[:, :, [3, 4, 5, 9, 10, 11]] *= length_torch.reshape(-1, 1, 1) 
#                 J_pred_T = J_pred.transpose(1, 2)
#                 task_e = task_error_torch
#                 approx_e = fps_vel_torch - torch.bmm(ends_vel_torch, J_pred_T)
#                 loss = self.mse_criterion(weight_approx_e * approx_e  + weight_task_e * task_e,   torch.zeros(approx_e.shape).to(self.device))

#                 loss.backward()
                
#                 # previous data stored in the sliding window
#                 if len(self.online_dataset) > 1:
#                     J_pred = self.model_J(state_input_batch_torch)
#                     J_pred[:, :, [3, 4, 5, 9, 10, 11]] *= length_batch_torch.reshape(-1, 1, 1)
#                     J_pred_T = J_pred.transpose(1, 2)
#                     loss =  self.lr_approx_e / window_size / 2 * self.mse_criterion(fps_vel_batch_torch, torch.bmm(ends_vel_batch_torch, J_pred_T))
                    
#                     loss.backward()

#                 # do the update
#                 self.online_optimizer.step()
#         # --------------------------------------------------

#         # store the data at the current time in the sliding window
#         self.online_dataset.append(state)
#         # remove the earliest data in the sliding window
#         if len(self.online_dataset) > window_size - 1:
#             self.online_dataset.pop(0)

#         # return the updated Jacobian matrix
#         J_pred = self.model_J(state_input_torch)
#         J_pred[:, :, [3, 4, 5, 9, 10, 11]] *= length_torch.reshape(-1, 1, 1)

#         return J_pred.cpu().detach().numpy().reshape(3 * self.numFPs, 12)


#     # --------------------------------------------------------------------
#     def calcNextEndsPose(self, current_ends_pose, ends_vel, delta_t=0.1):
#         left_end_pos = current_ends_pose[:, 0:3]
#         left_end_quat = current_ends_pose[:, 3:7]
#         right_end_pos = current_ends_pose[:, 7:10]
#         right_end_quat = current_ends_pose[:, 10:14]

#         left_end_lvel = ends_vel[:, 0:3]
#         left_end_avel = ends_vel[:, 3:6]
#         right_end_lvel = ends_vel[:, 6:9]
#         right_end_avel = ends_vel[:, 9:12]

#         next_left_end_pos = left_end_pos + left_end_lvel * delta_t
#         next_right_end_pos = right_end_pos + right_end_lvel * delta_t

#         left_end_ori = sciR.from_quat(left_end_quat)
#         left_end_delta_ori = sciR.from_rotvec(left_end_avel * delta_t)
#         next_left_end_ori = left_end_delta_ori * left_end_ori
#         next_left_end_quat = next_left_end_ori.as_quat()

#         right_end_ori = sciR.from_quat(right_end_quat)
#         right_end_delta_ori = sciR.from_rotvec(right_end_avel * delta_t)
#         next_right_end_ori = right_end_delta_ori * right_end_ori
#         next_right_end_quat = next_right_end_ori.as_quat()

#         next_ends_pose = np.concatenate([next_left_end_pos, next_left_end_quat, next_right_end_pos, next_right_end_quat], axis=1)
#         return next_ends_pose


#     # ------------------------------------------------------
#     def predNextFPsPositions(self, length, fps_pos, ends_pose, ends_vel, delta_t):
#         if fps_pos.ndim == 1 or ends_pose.ndim == 1 or ends_vel.ndim==1:
#             length = length.reshape(1, -1)
#             fps_pos = fps_pos.reshape(1, -1)
#             ends_pose = ends_pose.reshape(1, -1)
#             ends_vel = ends_vel.reshape(1, -1)

#         fps_vel_pred = self.predFPsVelocities(length, fps_pos, ends_pose, ends_vel)
#         next_fps_pos = fps_pos + delta_t * fps_vel_pred
#         return next_fps_pos


#     # ------------------------------------------------------
#     def predFPsVelocities(self, length, fps_pos, ends_pose, ends_vel):
#         state_input = np.concatenate([fps_pos, ends_pose], axis=1).astype('float32') # one row matrix
#         # np array to torch tensor
#         state_input_torch = self.relativeStateRepresentationTorch(torch.tensor(state_input)).to(self.device)
#         ends_vel_torch = torch.reshape(torch.tensor(ends_vel.astype('float32')), (-1, 1, 12)).to(self.device)
#         length_torch = torch.tensor(length.astype('float32')).to(self.device)

#         # predict the feature velocities
#         J_pred = self.model_J(state_input_torch)
#         J_pred[:, :, [3, 4, 5, 9, 10, 11]] *= length_torch.reshape(-1, 1, 1)

#         J_pred_T = J_pred.transpose(1, 2)
#         fps_vel_pred = torch.bmm(ends_vel_torch, J_pred_T).cpu().detach().numpy().reshape(-1, self.numFPs * 3)

#         return fps_vel_pred

#     # ------------------------------------------------------
#     # state representation preprocess
#     def relativeStateRepresentationTorch(self, state_input):
#         b_numpy = False
#         if type(state_input) is np.ndarray:
#             state_input = torch.tensor(state_input)
#             b_numpy = True

#         numFPs = self.numFPs
#         left_end_pos = state_input[:, 3*numFPs : 3*numFPs + 3]
#         left_end_quat = state_input[:,  3*numFPs + 3 : 3*numFPs +7]
#         right_end_pos = state_input[:,  3*numFPs + 7 : 3*numFPs +10]
#         right_end_quat = state_input[:,  3*numFPs +10 : 3*numFPs +14]

#         fps_pos = state_input[:, 0 : 3*numFPs].reshape(-1, numFPs, 3)
#         if len(fps_pos.shape) == 1: # reshape fps_pos from vector to one-row matrix
#             fps_pos = fps_pos.unsqueeze(0)

#         fps_pos_r = torch.zeros(fps_pos.shape)
#         fps_pos_r[:, 1:, :] = (fps_pos[:, 1:, :] - fps_pos[:, 0:-1, :]) 
#         fps_pos_r[:, 1:, :] /= torch.linalg.norm(fps_pos_r[:, 1:, :], dim=2).unsqueeze(2)
#         fps_pos_r = fps_pos_r.reshape(-1, 3*numFPs)
        
#         right_end_pos_r = (right_end_pos - left_end_pos) 
#         right_end_pos_r /= torch.linalg.norm(right_end_pos_r, dim=1).unsqueeze(1)

#         relative_state_input = torch.cat((fps_pos_r, right_end_pos_r, left_end_quat, right_end_quat), dim=1) #[60000,41], reduce input vector dimension by 3, combine q1,q2
#         if b_numpy:
#             return relative_state_input.numpy()
#         else:
#             return  relative_state_input


# # --------------------------------------------------------------------------------------------------------------------------------
# if __name__ == '__main__':
#     # project_dir = rospy.get_param("project_dir")
#     # env_dim = rospy.get_param("env/dimension")

#     project_dir = args.project_dir
#     env_dim = args.env_dimension

#     # training dataset (from 10 DLOs)
#     train_dataset = np.empty((0, I.state_dim)).astype("float32")
#     for j in range(1, 11):
#         # state [6000, 117]
#         # state = np.load(project_dir + "data/train_data/"+ env_dim + "/state_" + str(j) + ".npy").astype("float32")[: 6000, :]
#         state = np.load(os.path.join(project_dir, "data", "train_data", env_dim, f"state_{j}.npy")).astype("float32")[
#                 :6000, :]

#         train_dataset = np.concatenate([train_dataset, state], axis=0)

#     # train_dataset [60000,117]
#     #
#     # trainer = JacobianPredictor()
#     # trainer.LoadDataForTraining(train_dataset)
#     #
#     # trainer.Train(loadPreModel=False, n_epoch=100)

#    # test data, test for DLO 0
#     test_dataset = np.empty((0, I.state_dim)).astype("float32")
#     test_state = np.load(os.path.join(project_dir, "data", "train_data", env_dim, "state_0.npy")).astype("float32")[:, :]
#     test_dataset = np.concatenate([test_dataset, test_state], axis=0)
#     print(test_dataset.shape)
#     tester = JacobianPredictor()
#     tester.LoadDataForTest(test_dataset)
#     tester.TestAndSaveResults()



#!/home/xinge/miniconda3/envs/dlo/bin/python
# /usr/bin/env python
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

# import torch_rbf as rbf # reference: https://github.com/JeremyLinux/PyTorch-Radial-Basis-Function-Layer

from utils.data_augment import dataRandomTransform
from utils.state_index import I

import json

from torch_geometric.data import Data, Batch
import torch_geometric as pyg
from torch_geometric.nn import radius_graph
from GN_model_DLO import MySubEdge, MySubNode
from torch_geometric.nn import global_mean_pool

import argparse
parser = argparse.ArgumentParser(description="DLO Parser")


def load_args_from_file(filepath='/home/jessica/shape_control_DLO_2/ws_dlo/src/dlo_system_pkg/config/config_sim.json'):
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
tag = 'trial'


class Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


args_GN = Args(
    lr=1e-4
)


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------
#seperate the x, s, v, x_dot from the state vector (dim=117)
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


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter_add, scatter_mean
from torch_geometric.utils import to_dense_batch
from utils.utils_graph import get_graph_batch_RBF_comparable_fast


class MP1_NodeHead_JacobianNet(nn.Module):
    """
    1 message passing layer (manual) + node-specific heads for Ji.

    Expects PyG Batch `data` with:
      data.x         : [total_nodes, node_in]
      data.edge_index: [2, total_edges]
      data.edge_attr : [total_edges, edge_in]
      data.batch     : [total_nodes]   (graph id per node)
      data.u         : [num_graphs, u_dim]  (global, e.g., 11)

    Assumption:
      - Each graph has exactly numFPs nodes (FPS only), no end nodes.
    Output:
      J: [B, 3*numFPs, 12]
    """
    def __init__(
        self,
        numFPs: int,
        node_in: int,
        edge_in: int,
        u_dim: int = 11,
        hidden: int = 256,
        agg: str = "mean",   # "mean" or "sum"
    ):
        super().__init__()
        self.numFPs = numFPs
        self.node_in = node_in
        self.edge_in = edge_in
        self.u_dim = u_dim
        self.hidden = hidden
        self.agg = agg

        # --- A) edge MLP: edge_attr -> hidden (one layer) ---
        # self.edge_mlp = nn.Linear(edge_in, hidden, bias=False)

        self.edge_mlp  = nn.Sequential(
            nn.Linear(edge_in, hidden, bias=False),
            nn.ReLU(inplace=True)
            # nn.Linear(hidden, hidden,bias=False),
            # nn.ReLU(inplace=True),
        )

        # --- B) node MLP: x -> hidden (one layer) ---
        # self.node_mlp = nn.Linear(node_in, hidden, bias=False)

        self.node_mlp = nn.Sequential(
            nn.Linear(node_in, hidden, bias=False),
            nn.ReLU(inplace=True)
            # nn.Linear(hidden, hidden,bias=False),
            # nn.ReLU(inplace=True),
            )

        # --- C) fusion MLP: [node_h, msg_h, u] -> hidden ---
        self.fuse = nn.Sequential(
            nn.Linear(hidden+hidden, hidden, bias=False),
            nn.ReLU(inplace=True),
            # nn.Linear(hidden, hidden, bias=False),
            # nn.ReLU(inplace=True),
        )

        # # --- D) node-specific heads: hidden -> 36 (= 12*3) per node index i ---
        # self.heads = nn.ModuleList([nn.Linear(hidden, 36, bias=False) for _ in range(numFPs)])

        # --- D) shared head: hidden -> 36 (=12*3) for every node ---
        self.head = nn.Linear(hidden, 36, bias=False)

    def forward(self, data):
        if not hasattr(data, "u"):
            raise ValueError("data.u is required (global features), but missing.")
        if data.x.dim() != 2:
            raise ValueError(f"data.x must be [N, node_in], got {data.x.shape}")

        x = data.x                      # [Nt, node_in]
        edge_index = data.edge_index    # [2, Et]
        eattr = data.edge_attr          # [Et, edge_in]
        batch = data.batch              # [Nt]
        u = data.u                      # [B, u_dim]

        Nt = x.size(0)
        B = u.size(0)

        # 1) node embedding
        # node_h = F.relu(self.node_mlp(x), inplace=True)  # [Nt, hidden]

        node_h = self.node_mlp(x)
        # node_h = x

        # 2) edge message from edge_attr only (as you requested)
        # msg_e = F.relu(self.edge_mlp(eattr), inplace=True)  # [Et, hidden]

        msg_e = self.edge_mlp(eattr)

        # 3) aggregate messages to destination nodes
        src, dst = edge_index[0], edge_index[1]  # [Et], [Et]
        if self.agg == "sum":
            msg_h = scatter_add(msg_e, dst, dim=0, dim_size=Nt)   # [Nt, hidden]
        elif self.agg == "mean":
            msg_h = scatter_mean(msg_e, dst, dim=0, dim_size=Nt)  # [Nt, hidden]
        else:
            raise ValueError(f"Unknown agg='{self.agg}', use 'mean' or 'sum'.")

        # 5) fuse
        z = torch.cat([node_h, msg_h], dim=-1)        # [Nt, 2H+u_dim]
        # z = torch.cat([node_h], dim=-1)  # [Nt, 2H+u_dim]
        h = self.fuse(z)                                      # [Nt, hidden]
        # h = z

        # 6) reshape to [B, N, hidden]
        h_dense, mask = to_dense_batch(h, batch=batch, max_num_nodes=self.numFPs)  # [B,N,H]
        if h_dense.size(1) != self.numFPs:
            raise ValueError(f"Expected numFPs={self.numFPs} nodes per graph, got {h_dense.size(1)}")

        if not mask.all():
            raise ValueError("Found missing nodes in a graph (mask not all True).")

        # 7) shared head for all nodes (vectorized)
        out = self.head(h_dense)                     # [B, N, 36]

        # 8) convert to J: [B, 3N, 12]
        out = out.view(B, self.numFPs, 12, 3).transpose(2, 3)  # [B,N,3,12]
        J = out.reshape(B, 3 * self.numFPs, 12)                # [B,3N,12]
        return J


numFPs = args.DLO_num_FPs
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        
        self.bTrainMuBeta = True

        # self.model_J = MySimulator().to(self.device)

        self.model_J = MP1_NodeHead_JacobianNet(
            numFPs=10,
            node_in=19,
            edge_in=4+19+19,
            u_dim=11,
            hidden=256,
            agg='mean',
        ).to(self.device)

        print("Model J")
        print(self.model_J)

        self.optimizer = torch.optim.Adam(self.model_J.parameters(), lr=1e-3)

        # for offline learning
        # torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99, last_epoch=-1, verbose=False)
        self.criterion = torch.nn.SmoothL1Loss(reduction='mean', beta=0.001)
        # for online learning
        # self.online_optimizer = torch.optim.SGD([{'params': self.model_J.fc2.parameters()}], lr=1.0/self.online_update_rate)
        self.online_optimizer = torch.optim.SGD(self.model_J.head.parameters(), lr=1.0 / self.online_update_rate)
        self.mse_criterion = torch.nn.MSELoss(reduction='sum')

        # if rospy.get_param("learning/is_test"):
        if args.learning_is_test:
            # self.nnWeightDir = self.projectDir + 'ws_dlo/src/dlo_manipulation_pkg/models_test/rbfWeights/' + self.env_dim + '/'
            self.nnWeightDir = os.path.join(
                self.projectDir,
                'ws_dlo', 'src', 'dlo_manipulation_pkg', 'models_test',
                'gnnWeights', self.env_dim, ''
            )
        else:
            # self.nnWeightDir = self.projectDir + 'ws_dlo/src/dlo_manipulation_pkg/models/rbfWeights/' + self.env_dim + '/'
            self.nnWeightDir = os.path.join(
                self.projectDir,
                'ws_dlo', 'src', 'dlo_manipulation_pkg', 'models',
                'gnnWeights', self.env_dim, ''
            )
        # self.resultsDir = self.projectDir + 'results/' + self.env + '/'
        self.resultsDir = os.path.join(self.projectDir, 'results', self.env , '')
        # self.dataDir = self.projectDir +'data/'
        self.dataDir = os.path.join(self.projectDir , 'data', '')

        print('Directory')
        print(self.nnWeightDir)
        print(self.resultsDir)
        print(self.dataDir)

        self.online_dataset = []

        
    # ------------------------------------------------------
    def LoadDataForTraining(self, train_dataset=None):
        # trainset
        if train_dataset is None:
            train_dataset = np.load(self.dataDir + 'train_data/' + self.env_dim + '/state_0.npy').astype(np.float32)[600*2 : 600*10, :]
        self.trainDataset = NNDataset(train_dataset.astype(np.float32))
        self.trainDataLoader = DataLoader(self.trainDataset, batch_size=512, shuffle=True, num_workers=0)
        # self.trainDataLoader = DataLoader(self.trainDataset, batch_size=train_dataset.shape[0], shuffle=True, num_workers=0)

    # ------------------------------------------------------
    def LoadDataForTest(self, test_dataset=None):
        # testset
        if test_dataset is None:
            test_dataset = np.load(self.dataDir + 'train_data/' + self.env_dim + '/state_0.npy').astype(np.float32)[600*0 : 600*2, :]
        self.testDataset = NNDataset(test_dataset.astype(np.float32))
        self.testDataLoader = DataLoader(self.testDataset, batch_size=512, shuffle=False, num_workers=0)



    # ------------------------------------------------------
    def LoadModelWeights(self, file=None):
        offline_model = args.controller_offline_model  # 10*6 10*2 ...
        if file is not None:
            if os.path.exists(self.nnWeightDir + offline_model + "/" + file):
                print("Load Model path", self.nnWeightDir + offline_model + "/" + file)
                ckpt_path = os.path.join(self.nnWeightDir, offline_model, file)
                self.model_J.load_state_dict(torch.load(ckpt_path, map_location=self.device))

                print('Load previous model.', file)
            else:
                print('Warning: no model exists !')
        else:
            # offline_model = rospy.get_param("controller/offline_model")
            offline_model = args.controller_offline_model
            # if rospy.get_param("learning/is_test"):
            if args.learning_is_test:
                if os.path.exists(self.nnWeightDir  + "/model_J.pth"):
                    ckpt_path = os.path.join(self.nnWeightDir, "model_J.pth")
                    self.model_J.load_state_dict(torch.load(ckpt_path, map_location=self.device))

                    # print('Load previous model.')
                else:
                    print('Warning: no model exists !')
            else:
                if os.path.exists(self.nnWeightDir + offline_model + "/model_J.pth"):
                    ckpt_path = os.path.join(self.nnWeightDir, offline_model, "model_J.pth")
                    self.model_J.load_state_dict(torch.load(ckpt_path, map_location=self.device))

                    # print('Load previous model.')
                else:
                    print('Warning: no model exists !')

            self.n_count = 0
            self.online_dataset = []

    
    # ------------------------------------------------------
    def SaveModelWeights(self, path=""):
        # ensure both base dir and offline-model subdir exist
        os.makedirs(self.nnWeightDir, exist_ok=True)
        os.makedirs(os.path.join(self.nnWeightDir, args.controller_offline_model), exist_ok=True)

        if path:
            torch.save(self.model_J.state_dict(), path)
        else:
            torch.save(
                self.model_J.state_dict(),
                os.path.join(self.nnWeightDir, args.controller_offline_model, "model_J.pth")
            )
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

        log_dir = os.path.join(self.resultsDir, "logs")
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, f"loss_GNN_NN_{tag}.txt")
        with open(log_file, "w") as f:
            f.write("Epoch,Loss\n")


        # training
        loss_lst = []
        for epoch in range(0, n_epoch):
            accumLoss = 0.0
            numBatch = 0

            edge_cache = {}  # keep outside loop
            for batch_idx, (length, state_input, fps_vel, ends_vel) in enumerate(self.trainDataLoader):       
                # data augmentation
                if rotation_augmentation:
                    state_input, fps_vel, ends_vel = dataRandomTransform(state_input, fps_vel, ends_vel)
                # state_input = self.relativeStateRepresentationTorch(state_input)

                # construct graph

                # graph_lst = get_graph_batch(state_input, length, numFPs=args.DLO_num_FPs)
                # graph_batch = Batch.from_data_list(graph_lst)
                #
                # graph_batch = graph_batch.to(self.device)

                state_input = state_input.to(self.device)
                length = length.to(self.device)

                data, edge_cache = get_graph_batch_RBF_comparable_fast(
                    state_input, length, numFPs=self.numFPs,
                    edge_index_cache=edge_cache,
                )
                data = data.to(self.device)
                # data.x: [B*N, ...], data.edge_index: [2, B*E], data.batch: [B*N], data.u: [B,11]

                # print(graph_batch)

                # normalization
                ends_vel /= (torch.linalg.norm(fps_vel, dim=1).unsqueeze(1) + 1e-8)
                fps_vel /= (torch.linalg.norm(fps_vel, dim=1).unsqueeze(1) + 1e-8) # avoid division by zero

                # data to GPU
                length = length.to(self.device)
                state_input = state_input.to(self.device)
                fps_vel = fps_vel.to(self.device)
                ends_vel = ends_vel.to(self.device)

                bmm_ends_vel = torch.reshape(ends_vel, (-1, 1, 12))
                bmm_fps_vel = torch.reshape(fps_vel, (-1, 1, self.numFPs * 3))

                J_pred = self.model_J(data)                              # [B,3N,12]
                J_pred[:, :, [3, 4, 5, 9, 10, 11]] *= length.view(-1, 1, 1)

                J_pred_T = J_pred.transpose(1, 2)                          # [B,12,3N]
                loss = self.criterion(bmm_fps_vel, torch.bmm(bmm_ends_vel, J_pred_T))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                accumLoss += loss.item()
                numBatch += 1

                # torch.save(J_pred, 'J_data.pt')
                # print("Tensor saved as 'tensor_data.pt'")
            print("epoch: ", epoch, " , Loss/train: ", accumLoss/numBatch)
            epoch_loss = accumLoss/numBatch
            loss_lst.append(epoch_loss)

            # Log loss immediately
            with open(log_file, "a") as f:
                f.write(f"{epoch + 1},{epoch_loss:.6f}\n")

            if save_model and epoch%20==0:
                self.SaveModelWeights(os.path.join(self.nnWeightDir, args.controller_offline_model, f"model_J_GNN_{tag}_{epoch}.pth"))

        # save final model
        if save_model:
            self.SaveModelWeights()

        # with open("loss_RBF.txt", "w") as f:
        #     for loss in loss_lst:
        #         f.write(f"{loss}\n")
        f.close()



    # ------------------------------------------------------
    def TestAndSaveResults(self):

        self.LoadModelWeights(f"model_J.pth")

        accumLoss = 0.0
        numBatch = 0
        edge_cache = {}  # keep outside loop
        for batch_idx, (length, state_input, fps_vel, ends_vel) in enumerate(self.testDataLoader):
            # state_input = self.relativeStateRepresentationTorch(state_input)

            # construct graph

            # graph_lst = get_graph_batch(state_input, length, numFPs=args.DLO_num_FPs)
            # graph_batch = Batch.from_data_list(graph_lst)
            #
            # graph_batch = graph_batch.to(self.device)

            state_input = state_input.to(self.device)
            length = length.to(self.device)

            data, edge_cache = get_graph_batch_RBF_comparable_fast(
                state_input, length, numFPs=self.numFPs,
                edge_index_cache=edge_cache,
            )

            length = length.to(self.device)
            state_input = state_input.to(self.device)
            fps_vel = fps_vel.to(self.device)
            ends_vel = ends_vel.to(self.device)

            bmm_ends_vel = torch.reshape(ends_vel, (-1, 1, 12))
            bmm_fps_vel = torch.reshape(fps_vel, (-1, 1, self.numFPs * 3))

            # J_pred = self.model_J(state_input)

            # J_pred = torch.reshape(torch.reshape(self.model_J(graph_batch), (-1, args.DLO_num_FPs, 12, 3)).transpose(2, 3),
            #                        (-1, 3 * args.DLO_num_FPs, 12))

            J_pred = self.model_J(data)  # [B,3N,12]

            J_pred[:, :, [3, 4, 5, 9, 10, 11]] *= length.reshape(-1, 1, 1)

            J_pred_T = J_pred.transpose(1, 2)
            fps_vel_pred = torch.bmm(bmm_ends_vel, J_pred_T)

            # testLoss = self.mse_criterion(fps_vel_pred, bmm_fps_vel)
            # testLoss = self.criterion(fps_vel_pred, bmm_fps_vel)

            eps = 1e-8  # avoid division by zero

            error = fps_vel_pred - bmm_fps_vel

            rel_l2_error = (
                    torch.norm(error, p=2, dim=-1) /
                    (torch.norm(bmm_fps_vel, p=2, dim=-1)+eps)
            )

            # average over batch / time / nodes
            testLoss = rel_l2_error.sum() * 100.0

            accumLoss += testLoss.item()
            numBatch += 1

        # print("Loss/test: ", accumLoss/numBatch)
        print("Loss/test: ", accumLoss/len(self.testDataset))

        # test result 数据保存
        np.save(self.resultsDir + "nn_test/rbf/" + self.env_dim + "/dot_x_truth.npy", bmm_fps_vel.cpu().detach().numpy()) 
        np.save(self.resultsDir + "nn_test/rbf/" + self.env_dim + "/dot_x_pred.npy", fps_vel_pred.cpu().detach().numpy())


    # ------------------------------------------------------
    def OnlineLearningAndPredictJ(self, state, task_error=None):
        state = copy.copy(state)
        task_error = copy.copy(task_error)
        
        self.n_count  += 1
        # parameters
        window_size = params_online_window_time * self.control_rate

        length = state[I.length_idx]
        state_input = state[I.state_input_idx].reshape(1, -1) # one row matrix
        if task_error is None:
            task_error = np.zeros((self.numFPs * 3, ), dtype='float32')

        # Because of the imperfection of the simulator, sometimes the DLO will wiggle to the other side very fast.
        # We don't want to include these outlier data in training, so we just discard the online data with too fast speed.
        fps_vel_norm = np.linalg.norm(state[I.fps_vel_idx])
        if fps_vel_norm > params_online_max_valid_fps_vel or  fps_vel_norm < params_online_min_valid_fps_vel:
            # return the Jacobian without online updating
            length_torch = torch.tensor(length).to(self.device)
            state_input_torch = self.relativeStateRepresentationTorch(torch.tensor(state_input)).to(self.device)
            J_pred = self.model_J(state_input_torch)
            J_pred[:, :, [3, 4, 5, 9, 10, 11]] *= length_torch.reshape(-1, 1, 1)  # no normalize 要改 # RBF_abs 要改
            return J_pred.cpu().detach().numpy().reshape(3 * self.numFPs, 12)

        # normalize the velocities
        elif fps_vel_norm > params_online_fps_vel_thres:
            state[I.fps_vel_idx]  /= (fps_vel_norm)
            state[I.ends_vel_idx]  /= (fps_vel_norm)
            task_error *= (fps_vel_norm)
        else:
            state[I.fps_vel_idx] /= params_online_fps_vel_thres
            state[I.ends_vel_idx]  /= params_online_fps_vel_thres
            task_error *= (params_online_fps_vel_thres) 

        fps_vel = state[I.fps_vel_idx]
        ends_vel = state[I.ends_vel_idx]
            
        # --------------------------------------------------
        # update the NN weights
        # we use the SGD optimizer in PyTorch for online learning implementation to achieve faster computing speed. 
        # Note that the following computing is mathematically equivalent to the online updating law in the paper.

        # learning rate: transform the learning rates  to the weights in the loss function
        weight_approx_e = np.sqrt(self.lr_approx_e /  window_size / 2)
        if weight_approx_e == 0:
            weight_task_e = 0
        else:
            weight_task_e = self.lr_task_e / 2 /  weight_approx_e

        # data preparation
        # latest step data
        length_torch = torch.tensor(length).to(self.device)
        state_input_torch = self.relativeStateRepresentationTorch(torch.tensor(state_input)).to(self.device)
        ends_vel_torch = torch.reshape(torch.tensor(ends_vel), (1, 1, 12)).to(self.device)
        fps_vel_torch = torch.reshape(torch.tensor(fps_vel), (1, 1, 3 * self.numFPs)).to(self.device)
        task_error_torch = torch.reshape(torch.tensor(task_error), (1, 1, 3 * self.numFPs)).to(self.device)

        # previous data in sliding window
        if len(self.online_dataset) > 1:
            online_dataset = np.array(self.online_dataset)
            length_batch = online_dataset[:, I.length_idx]
            state_input_batch = online_dataset[:, I.state_input_idx]
            if len(state_input_batch.shape) == 1:
                state_input_batch = state_input_batch.reshape(1, -1)
            fps_vel_batch = online_dataset[:, I.fps_vel_idx]
            ends_vel_batch = online_dataset[:, I.ends_vel_idx]
            length_batch_torch = torch.tensor(length_batch).to(self.device)
            state_input_batch_torch = self.relativeStateRepresentationTorch(torch.tensor(state_input_batch)).to(self.device)
            ends_vel_batch_torch = torch.reshape(torch.tensor(ends_vel_batch), (-1, 1, 12)).to(self.device)
            fps_vel_batch_torch = torch.reshape(torch.tensor(fps_vel_batch), (-1, 1, 3 * self.numFPs)).to(self.device)

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
                loss = self.mse_criterion(weight_approx_e * approx_e  + weight_task_e * task_e,   torch.zeros(approx_e.shape).to(self.device))

                loss.backward()
                
                # previous data stored in the sliding window
                if len(self.online_dataset) > 1:
                    J_pred = self.model_J(state_input_batch_torch)
                    J_pred[:, :, [3, 4, 5, 9, 10, 11]] *= length_batch_torch.reshape(-1, 1, 1)
                    J_pred_T = J_pred.transpose(1, 2)
                    loss =  self.lr_approx_e / window_size / 2 * self.mse_criterion(fps_vel_batch_torch, torch.bmm(ends_vel_batch_torch, J_pred_T))
                    
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

        next_ends_pose = np.concatenate([next_left_end_pos, next_left_end_quat, next_right_end_pos, next_right_end_quat], axis=1)
        return next_ends_pose



    # ------------------------------------------------------
    def predNextFPsPositions(self, length, fps_pos, ends_pose, ends_vel, delta_t):
        if fps_pos.ndim == 1 or ends_pose.ndim == 1 or ends_vel.ndim==1:
            length = length.reshape(1, -1)
            fps_pos = fps_pos.reshape(1, -1)
            ends_pose = ends_pose.reshape(1, -1)
            ends_vel = ends_vel.reshape(1, -1)

        fps_vel_pred = self.predFPsVelocities(length, fps_pos, ends_pose, ends_vel)
        next_fps_pos = fps_pos + delta_t * fps_vel_pred
        return next_fps_pos



    # ------------------------------------------------------
    def predFPsVelocities(self, length, fps_pos, ends_pose, ends_vel):
        state_input = np.concatenate([fps_pos, ends_pose], axis=1).astype('float32') # one row matrix
        # np array to torch tensor
        state_input_torch = self.relativeStateRepresentationTorch(torch.tensor(state_input)).to(self.device)
        ends_vel_torch = torch.reshape(torch.tensor(ends_vel.astype('float32')), (-1, 1, 12)).to(self.device)
        length_torch = torch.tensor(length.astype('float32')).to(self.device)

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
        left_end_pos = state_input[:, 3*numFPs : 3*numFPs + 3]
        left_end_quat = state_input[:,  3*numFPs + 3 : 3*numFPs +7]
        right_end_pos = state_input[:,  3*numFPs + 7 : 3*numFPs +10]
        right_end_quat = state_input[:,  3*numFPs +10 : 3*numFPs +14]

        fps_pos = state_input[:, 0 : 3*numFPs].reshape(-1, numFPs, 3)
        if len(fps_pos.shape) == 1: # reshape fps_pos from vector to one-row matrix
            fps_pos = fps_pos.unsqueeze(0)

        fps_pos_r = torch.zeros(fps_pos.shape)
        fps_pos_r[:, 1:, :] = (fps_pos[:, 1:, :] - fps_pos[:, 0:-1, :]) 
        fps_pos_r[:, 1:, :] /= torch.linalg.norm(fps_pos_r[:, 1:, :], dim=2).unsqueeze(2)
        fps_pos_r = fps_pos_r.reshape(-1, 3*numFPs)
        
        right_end_pos_r = (right_end_pos - left_end_pos) 
        right_end_pos_r /= torch.linalg.norm(right_end_pos_r, dim=1).unsqueeze(1)

        relative_state_input = torch.cat((fps_pos_r, right_end_pos_r, left_end_quat, right_end_quat), dim=1) #[60000,41], reduce input vector dimension by 3, combine q1,q2
        if b_numpy:
            return relative_state_input.numpy()
        else:
            return  relative_state_input


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

    print("Train dataset shape:", train_dataset.shape)

    # train_dataset [60000,117]
    trainer = JacobianPredictor()
    trainer.LoadDataForTraining(train_dataset)
    trainer.Train(loadPreModel=False, n_epoch=100, save_model=True, rotation_augmentation=True)

    print("Training finished. Saved to:",
          os.path.join(trainer.nnWeightDir, args.controller_offline_model, "model_J.pth"))
