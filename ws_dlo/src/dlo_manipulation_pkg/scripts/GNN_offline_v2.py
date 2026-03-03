# GNN_no_ros_new_v2.py  (no random occlusion inside)

import os, json, copy, numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from scipy.spatial.transform import Rotation as sciR

from torch_geometric.data import Data, Batch
from torch_geometric.nn import radius_graph, global_mean_pool

from utils.state_index import I
from GN_model_DLO import MySubEdge, MySubNode

import argparse
import rospy

# ------------------------------------------------------------------
# Load config
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Jacobian GNN v2 (masked input)")
parser.add_argument("--config", type=str,
                    default="/home/jessica/shape_control_DLO_2/ws_dlo/src/dlo_system_pkg/config/config.json")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--n_epoch", type=int, default=50)
parser.add_argument("--masked_weight", type=float, default=0.5)
args_cli, _ = parser.parse_known_args()


def load_args_from_file(filepath):
    with open(filepath, "r") as f:
        d = json.load(f)
    return argparse.Namespace(**d)

args = load_args_from_file(args_cli.config)

class ArgsGN:
    def __init__(self, lr=1e-4):
        self.lr = lr

args_GN = ArgsGN(lr=1e-4)

# ------------------------------------------------------------------
# Dataset (v2 expects preprocessed arrays, including fp_visible_mask)
# ------------------------------------------------------------------
class NNDatasetV2(Dataset):
    """
    state: full state or prepacked array with:
      - length
      - state_input (3*numFPs+14)
      - fps_vel
      - ends_vel
      - fp_visible_mask (numFPs)
    You will later create such an array in preprocessing.
    """
    def __init__(self, arr, numFPs):
        self.data_num = arr.shape[0]
        self.numFPs = numFPs

        self.length = arr[:, I.length_idx]
        self.state_input = arr[:, I.state_input_idx]
        self.fps_vel = arr[:, I.fps_vel_idx]
        self.ends_vel = arr[:, I.ends_vel_idx]

        # Here we assume fp_visible_mask is stored separately; for now
        # we point to indices after I.*; later you will define exact layout.
        # Placeholder: last numFPs entries in array:
        self.fp_visible_mask = arr[:, -self.numFPs:]  # adjust in preprocess

    def __getitem__(self, idx):
        return (self.length[idx],
                self.state_input[idx],
                self.fps_vel[idx],
                self.ends_vel[idx],
                self.fp_visible_mask[idx])

    def __len__(self):
        return self.data_num

# ------------------------------------------------------------------
# Graph builder with mask channel
# ------------------------------------------------------------------
def get_graph_data(state_input_1xD, length_scalar, fp_visible_mask_1xN, numFPs):
    """
    state_input_1xD: (1, 3*numFPs+14)
    fp_visible_mask_1xN: (1, numFPs)
    length_scalar: scalar tensor or float
    """
    left_end_pos = state_input_1xD[:, 3*numFPs:3*numFPs+3]
    left_end_quat = state_input_1xD[:, 3*numFPs+3:3*numFPs+7]
    right_end_pos = state_input_1xD[:, 3*numFPs+7:3*numFPs+10]
    right_end_quat = state_input_1xD[:, 3*numFPs+10:3*numFPs+14]
    fps_pos = state_input_1xD[:, 0:3*numFPs].reshape(numFPs, 3)

    left_end_state = torch.cat((left_end_pos, left_end_quat), dim=-1)    # (1,7)
    right_end_state = torch.cat((right_end_pos, right_end_quat), dim=-1) # (1,7)
    fps_state = torch.cat(
        (fps_pos, torch.zeros((numFPs, 4),
                              dtype=fps_pos.dtype,
                              device=fps_pos.device)),
        dim=-1
    )  # (numFPs,7)

    node_features_7 = torch.cat((left_end_state, fps_state, right_end_state), dim=0)  # (numFPs+2,7)

    # mask channel: ends always 1, FPs from fp_visible_mask
    mask_left = torch.ones((1, 1), device=node_features_7.device, dtype=node_features_7.dtype)
    mask_right = torch.ones((1, 1), device=node_features_7.device, dtype=node_features_7.dtype)
    mask_fps = fp_visible_mask_1xN.reshape(numFPs, 1).to(node_features_7.dtype)
    mask_channel = torch.cat((mask_left, mask_fps, mask_right), dim=0)  # (numFPs+2,1)

    node_input = torch.cat((node_features_7, mask_channel), dim=1)      # (numFPs+2,8)
    all_pos_xyz = node_input[:, :3]

    if isinstance(length_scalar, torch.Tensor):
        r = float(length_scalar.item()) 
    else:
        r = float(length_scalar) 

    edge_index = radius_graph(all_pos_xyz, r=r, loop=False)
    edge_attr = all_pos_xyz[edge_index[0]] - all_pos_xyz[edge_index[1]]

    return Data(x=node_input, edge_index=edge_index, edge_attr=edge_attr, allposxyz=all_pos_xyz)

def get_graph_batch(state_input_BxD, length_B, fp_visible_mask_BxN, numFPs):
    graphs = []
    B = state_input_BxD.shape[0]
    length_B = length_B.view(-1)
    for i in range(B):
        graphs.append(
            get_graph_data(state_input_BxD[i:i+1, :],
                           length_B[i],
                           fp_visible_mask_BxN[i:i+1, :],
                           numFPs=numFPs)
        )
    return graphs

# ------------------------------------------------------------------
# Model (node_dim = 8)
# ------------------------------------------------------------------
class MySimulator(nn.Module):
    def __init__(self):
        super().__init__()
        self.args = args_GN
        self.node_dim = 8
        self.edge_dim = 3
        self.hidden_dim = 256
        self.output_dim = 360  # 3*numFPs*12 for numFPs=10

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

        self.subnets = nn.ModuleList([
            subnet0_ee, subnet1_ee, subnet2_ee,
            subnet3_en, subnet4_en,
            subnet5_pe, subnet6_pn,
            subnet7_pe, subnet8_pn,
            subnet9_pe, subnet10_pn
        ])

    def forward(self, data):
        node_feature = data.x
        edge_feature = data.edge_attr
        edge_index = data.edge_index
        batch = data.batch

        for subnet in self.subnets:
            if subnet.type == 'ee':
                node_out, edge_feature = subnet.forward(node_feature, node_feature, edge_index, edge_feature)
            elif subnet.type == 'en':
                node_out, node_feature = subnet.forward(node_feature)
            elif subnet.type == 'pe':
                node_out, edge_feature, input_to_node = subnet.forward(node_feature, node_feature, edge_index, edge_feature)
            elif subnet.type == 'pn':
                node_out, node_feature = subnet.forward(input_to_node)

        graph_repr = global_mean_pool(node_feature, batch)
        out = self.output_layer(graph_repr)
        return out

# ------------------------------------------------------------------
# JacobianPredictor v2 (no internal occlusion)
# ------------------------------------------------------------------
class JacobianPredictor:
    def __init__(self):
        self.numFPs = args.DLO_num_FPs
        self.projectDir = args.project_dir
        self.env = args.env_sim_or_real
        self.env_dim = args.env_dimension

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_J = MySimulator().to(self.device)
        self.optimizer = torch.optim.Adam(self.model_J.parameters(), lr=1e-3)

        self.smoothl1 = torch.nn.SmoothL1Loss(reduction="none", beta=0.001)

        if args.learning_is_test:
            self.nnWeightDir = os.path.join(
                self.projectDir, "ws_dlo", "src", "dlo_manipulation_pkg",
                "models_test", "gnnWeights", self.env_dim, ""
            )
        else:
            self.nnWeightDir = os.path.join(
                self.projectDir, "ws_dlo", "src", "dlo_manipulation_pkg",
                "models", "gnnWeights", self.env_dim, ""
            )

        self.resultsDir = os.path.join(self.projectDir, "results", self.env, "")
        self.dataDir = os.path.join(self.projectDir, "data", "")

        os.makedirs(self.nnWeightDir, exist_ok=True)

    def load_data_for_training(self, train_array):
        self.trainDataset = NNDatasetV2(train_array.astype(np.float32), numFPs=self.numFPs)
        self.trainLoader = DataLoader(
            self.trainDataset,
            batch_size=args_cli.batch_size,
            shuffle=True,
            num_workers=args_cli.num_workers
        )

    def save_model(self, name="model_J.pth"):
        torch.save(self.model_J.state_dict(), os.path.join(self.nnWeightDir, name))

    def train(self):
        log_dir = getattr(args, "losslogfolder",
                          os.path.join(self.projectDir, "results", "losslogs"))
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "loss_GNN_v2_masked.txt")
        with open(log_file, "w") as f:
            f.write("Epoch,Loss\n")

        for epoch in range(args_cli.n_epoch):
            self.model_J.train()
            accum = 0.0
            nb = 0

            for (length, state_input, fps_vel, ends_vel, fp_mask) in self.trainLoader:
                length = length.to(self.device).float().view(-1)
                state_input = state_input.to(self.device).float()
                fps_vel = fps_vel.to(self.device).float()
                ends_vel = ends_vel.to(self.device).float()
                fp_mask = fp_mask.to(self.device).float()  # (B,numFPs)

                B = state_input.shape[0]

                # Graphs
                graphs = get_graph_batch(state_input, length, fp_mask, numFPs=self.numFPs)
                graph_batch = Batch.from_data_list(graphs).to(self.device)

                # Normalize velocities as in original code
                norm = torch.linalg.norm(fps_vel, dim=1).unsqueeze(1) + 1e-8
                ends_vel = ends_vel / norm
                fps_vel = fps_vel / norm

                bmm_ends_vel = ends_vel.view(-1, 1, 12)
                bmm_fps_vel = fps_vel.view(-1, 1, self.numFPs * 3)

                # Forward
                Jflat = self.model_J(graph_batch)  # (B,360)
                J_pred = torch.reshape(
                    torch.reshape(Jflat, (-1, self.numFPs, 12, 3)).transpose(2, 3),
                    (-1, 3 * self.numFPs, 12)
                )
                J_pred[:, :, [3, 4, 5, 9, 10, 11]] *= length.view(-1, 1, 1)  # length scaling

                J_pred_T = J_pred.transpose(1, 2)
                fps_vel_pred = torch.bmm(bmm_ends_vel, J_pred_T)  # (B,1,3*numFPs)

                # ---- mask-weighted loss ----
                w = fp_mask * 1.0 + (1.0 - fp_mask) * float(args_cli.masked_weight)  # (B,numFPs)
                w3 = w.repeat_interleave(3, dim=1).view(B, 1, 3 * self.numFPs)

                elem = self.smoothl1(fps_vel_pred, bmm_fps_vel)  # (B,1,3*numFPs)
                loss = (elem * w3).mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                accum += float(loss.item())
                nb += 1

            epoch_loss = accum / max(nb, 1)
            print(f"Epoch {epoch:03d} loss {epoch_loss:.6f}")
            with open(log_file, "a") as f:
                f.write(f"{epoch+1},{epoch_loss:.8f}\n")

        self.save_model()
    
    def LoadModelWeights(self, file=None):
        # folder name from JSON
        model_folder = args.controller_offline_model   # "gnn_occluded_v1"

        # default filename inside that folder
        if file is None:
            file = "model_J.pth"

        # if caller passes absolute file path, use it directly
        if os.path.isabs(file):
            wpath = file
        else:
            wpath = os.path.join(self.nnWeightDir, model_folder, file)

        print("Loading GNN weights from:", wpath)

        if not os.path.exists(wpath):
            raise FileNotFoundError(f"GNN weights not found: {wpath}")

        sd = torch.load(wpath, map_location=self.device)
        self.model_J.load_state_dict(sd, strict=True)
        self.model_J = self.model_J.to(self.device).float()
        self.model_J.eval()

    @torch.no_grad()
    def PredictJacobian(self, state, fp_visible_mask):
        """
        state: full env state vector (numpy)
        fp_visible_mask: (numFPs,) with 1 visible, 0 occluded (float or bool). [file:4]
        returns: Jacobian (3*numFPs, 12) numpy float32
        """
        length = float(state[I.length_idx])
        state_input = state[I.state_input_idx].reshape(1, -1).astype(np.float32)
        fp_visible_mask = np.asarray(fp_visible_mask, dtype=np.float32).reshape(1, self.numFPs)

        length_t = torch.tensor([length], dtype=torch.float32, device=self.device).view(-1)
        state_input_t = torch.tensor(state_input, dtype=torch.float32, device=self.device)
        mask_t = torch.tensor(fp_visible_mask, dtype=torch.float32, device=self.device)

        graphs = get_graph_batch(state_input_t, length_t, mask_t, numFPs=self.numFPs)  # your existing fn
        graph_batch = Batch.from_data_list(graphs).to(self.device)

        Jflat = self.model_J(graph_batch)  # (1, 360) when numFPs=10 [file:4]
        J_pred = torch.reshape(
            torch.reshape(Jflat, (-1, self.numFPs, 12, 3)).transpose(2, 3),
            (-1, 3 * self.numFPs, 12)
        )
        J_pred[:, :, [3, 4, 5, 9, 10, 11]] *= length_t.view(-1, 1, 1)

        return J_pred[0].detach().cpu().numpy().astype(np.float32)


# ------------------------------------------------------------------
# Main (for now, expects you to pass a preprocessed array)
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Placeholder: load a preprocessed .npy that you will create later.
    # Example path: project_dir/data/train_data/{env_dim}/state_preprocessed_v2.npy
    # train_pre_path = os.path.join(
    #     args.projectdir, "data", "train_data", args.envdimension, "state_preprocessed_v2.npy"
    # )
    # train_pre_path="/home/jessica/shape_control_DLO_2/data/train_data/2D"
    # train_array = np.load(train_pre_path).astype("float32")

    project_dir = args.project_dir
    env_dim = args.env_dimension

    train_dataset = np.empty((0, I.state_dim_v2)).astype("float32")
    for j in range(1, 11):
        # state [6000, 117]
        # state = np.load(project_dir + "data/train_data/"+ env_dim + "/state_" + str(j) + ".npy").astype("float32")[: 6000, :]
        state = np.load(os.path.join(project_dir, "data", "train_data", env_dim, f"state_{j}_mask_1.npy")).astype("float32")[
            :6000, :]

        train_dataset = np.concatenate([train_dataset, state], axis=0)

      # train_dataset [60000,117]


    trainer = JacobianPredictor()
    trainer.load_data_for_training(train_dataset)
    trainer.train()
