import torch
import numpy as np

# ---------------------------------------------------------------------
# Global device (cuda if available else cpu)
# ---------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_graph_features(G, action, state, delta_state, bs=1, norm=True, noise=0.003, std=None):

    pos = state[:, 5:5+18].view(-1, 6, 3)
    if noise > 0:
        pos_noise = torch.randn(pos.size(), device=DEVICE) * noise * std[:, :3].to(DEVICE)
    else:
        pos_noise = 0

    pos = pos + pos_noise
    if noise > 0:
        delta_state[:, 5:5+18] -= pos_noise.view(-1, 18)

    # differences in the two body pos angle (w)
    # same as retrieving joint angle from state directly state[:, :5]
    joints = pos[:, 1:, -1] - pos[:, :-1, -1]
    joints[joints > np.pi] -= np.pi * 2
    joints[joints < -np.pi] += np.pi * 2

    if norm:
        center_pos = torch.mean(pos[:, :, :2], dim=1, keepdim=True)
        pos[:, :, :2] -= center_pos

    vel = state[:, 5+18:5+36].view(-1, 6, 3)

    if noise > 0:
        vel_noise = torch.randn(vel.size(), device=DEVICE) * noise * std[:, 3:].to(DEVICE)
    else:
        vel_noise = 0
    vel = vel + vel_noise

    if noise > 0:
        delta_state[:, 5+18:5+36] -= vel_noise.view(-1, 18)

    for node in G.nodes():
        #print(node)
        G.nodes[node]['feat'][:, :3] = pos[:, node]
        G.nodes[node]['feat'][:, 3:] = vel[:, node]

    for edge in G.edges():
        if edge[0] < edge[1]:
            G[edge[0]][edge[1]]['feat'][:, 0] = -1
        else:
            G[edge[0]][edge[1]]['feat'][:, 0] = 1

        m = min(edge)
        G[edge[0]][edge[1]]['feat'][:, 1] = joints[:, m]
        G[edge[0]][edge[1]]['feat'][:, 2] = action[:, m]
    return G


def build_graph_loss(G, state):
    loss = 0
    n_nodes = len(G)

    pos = state[:, 5:5 + 18].view(-1, 6, 3)
    pos[:, :, 2] -= (pos[:, :, 2] > np.pi).float() * np.pi * 2
    pos[:, :, 2] += (pos[:, :, 2] < -np.pi).float() * np.pi * 2

    vel = state[:, 5 + 18:5 + 36].view(-1, 6, 3)

    for node in G.nodes():
        loss += torch.mean((G.nodes[node]['feat'][:, :3] - pos[:, node]) ** 2)
        loss += torch.mean((G.nodes[node]['feat'][:, 3:] - vel[:, node]) ** 2)

    loss /= n_nodes
    return loss


def build_graph_loss2(G, H):
    loss = 0
    n_nodes = len(G)
    for node in G.nodes():
        loss += torch.mean((G.nodes[node]['feat'][:, :3] - H.nodes[node]['feat'][:, :3]) ** 2)
        loss += torch.mean((G.nodes[node]['feat'][:, 3:] - H.nodes[node]['feat'][:, 3:]) ** 2)

    loss /= n_nodes
    return loss


def init_graph_features(G, graph_feat_size, node_feat_size, edge_feat_size, bs=1, cuda=False):
    # Keep original signature/behavior; route to DEVICE if cuda=True
    if cuda:
        G.graph['feat'] = torch.zeros(bs, graph_feat_size, device=DEVICE)
        for node in G.nodes():
            G.nodes[node]['feat'] = torch.zeros(bs, node_feat_size, device=DEVICE)
        for edge in G.edges():
            G[edge[0]][edge[1]]['feat'] = torch.zeros(bs, edge_feat_size, device=DEVICE)
    else:
        G.graph['feat'] = torch.zeros(bs, graph_feat_size)
        for node in G.nodes():
            G.nodes[node]['feat'] = torch.zeros(bs, node_feat_size)
        for edge in G.edges():
            G[edge[0]][edge[1]]['feat'] = torch.zeros(bs, edge_feat_size)


def detach(G):
    G.graph['feat'] = G.graph['feat'].detach()
    for node in G.nodes():
        G.nodes[node]['feat'] = G.nodes[node]['feat'].detach()
    for edge in G.edges():
        G[edge[0]][edge[1]]['feat'] = G[edge[0]][edge[1]]['feat'].detach()
    return G


import torch
import torch_geometric as pyg

# -----------------------------
# Precompute edge_index template ONCE (N fixed)
# -----------------------------
def _fully_connected_edge_index(N: int, device, bidirectional=True, self_loops=False):
    idx = torch.arange(N, device=device)
    src = idx.repeat_interleave(N)
    dst = idx.repeat(N)
    if not self_loops:
        mask = src != dst
        src, dst = src[mask], dst[mask]
    ei = torch.stack([src, dst], dim=0)
    if not bidirectional:
        # for FC, this already includes both directions (i->j and j->i) when self_loops=False
        pass
    return ei  # [2, E]


def _chain_edge_index(N: int, device, bidirectional=True):
    src = torch.arange(N - 1, device=device)
    dst = src + 1
    ei = torch.stack([src, dst], dim=0)  # forward
    if bidirectional:
        ei_rev = torch.stack([dst, src], dim=0)
        ei = torch.cat([ei, ei_rev], dim=1)
    return ei  # [2, E]


def build_present_mask_from_missing(missing_list, numFPs, device):
    B = len(missing_list)
    present = torch.ones((B, numFPs), dtype=torch.bool, device=device)
    for b, miss in enumerate(missing_list):
        if len(miss) > 0:
            present[b, torch.tensor(miss, device=device, dtype=torch.long)] = False
    return present


# new formulation
def get_graph_batch_RBF_comparable_fast(
    state_input: torch.Tensor,   # [B, D]
    length: torch.Tensor,        # [B] or [B,1]
    numFPs: int,
       missing_lst=None,
    use_length_norm: bool = True,
    use_left_anchor: bool = True,
    add_distance_to_edge: bool = True,
    include_quat_global: bool = True,   # u dim = 11 else 3
    use_chain_edges: bool = False,      # FC is expensive
    append_u_to_x: bool = True,
    eps: float = 1e-8,
    edge_index_cache: dict = None,      # optional: pass a dict to cache templates
):
    """
    Vectorized batch graph construction (FAST).
    Returns a single pyg.data.Data with B graphs batched.
    """
    if state_input.dim() != 2:
        raise ValueError(f"state_input must be [B,D], got {state_input.shape}")

    # Ensure inputs live on the global DEVICE (cuda if available else cpu)
    state_input = state_input.to(DEVICE)
    length = length.to(DEVICE)

    device = state_input.device
    B = state_input.size(0)

    if missing_lst is None:
        present_mask = torch.ones((B, numFPs), dtype=torch.bool, device=device)
    else:
        present_mask = build_present_mask_from_missing(missing_lst, numFPs=10, device=device)
        # shape: [B, 10]
        print("present_mask", present_mask)

    length = length.view(B, 1).to(device=device)
    length_safe = length + eps

    # ---------------- Parse state ----------------
    fps_pos = state_input[:, 0:3*numFPs].reshape(B, numFPs, 3)           # [B,N,3]
    left_pos = state_input[:, 3*numFPs : 3*numFPs + 3]                  # [B,3]
    left_q   = state_input[:, 3*numFPs + 3 : 3*numFPs + 7]              # [B,4]
    right_pos = state_input[:, 3*numFPs + 7 : 3*numFPs + 10]            # [B,3]
    right_q   = state_input[:, 3*numFPs + 10 : 3*numFPs + 14]           # [B,4]

    if use_left_anchor:
        anchor = left_pos
        left_pos  = left_pos - anchor
        right_pos = right_pos - anchor
        fps_pos   = fps_pos - anchor.unsqueeze(1)

    # end2end axis
    end_vec = right_pos - left_pos                                      # [B,3]
    end_len = torch.norm(end_vec, dim=-1, keepdim=True) + eps           # [B,1]
    end2end = end_vec / end_len                                         # [B,3]

    # ---------------- Global u ----------------
    if include_quat_global:
        u = torch.cat([end2end, left_q, right_q], dim=1)                # [B,11]
    else:
        u = end2end                                                     # [B,3]
    u_dim = u.size(1)

    # ---------------- Node features ----------------
    # precompute s once (cheap but do it vectorized)
    s = torch.linspace(0.0, 1.0, steps=numFPs, device=device).view(1, numFPs, 1).expand(B, -1, -1)  # [B,N,1]

    dL = (fps_pos - left_pos.unsqueeze(1)) / end_len.unsqueeze(1)       # [B,N,3]
    dR = (fps_pos - right_pos.unsqueeze(1)) / end_len.unsqueeze(1)      # [B,N,3]

    t = torch.sum((fps_pos - left_pos.unsqueeze(1)) * end2end.unsqueeze(1), dim=-1, keepdim=True)   # [B,N,1]
    t = t / end_len.unsqueeze(1)

    proj_vec = (t * end_len.unsqueeze(1)) * end2end.unsqueeze(1)        # [B,N,3]
    perp = (fps_pos - left_pos.unsqueeze(1)) - proj_vec                 # [B,N,3]
    d_perp = torch.norm(perp, dim=-1, keepdim=True) / end_len.unsqueeze(1)  # [B,N,1]

    x = torch.cat([dL, dR, t, d_perp], dim=-1)                       # [B,N,9]

    if append_u_to_x:
        x = torch.cat([x, u.unsqueeze(1).expand(B, numFPs, u_dim)], dim=-1)  # [B,N,9+u_dim]

    # flatten node features
    x_flat = x.reshape(B * numFPs, -1)                                  # [B*N, x_dim]

    # ---------------- Edge index template (cache) ----------------
    if edge_index_cache is None:
        edge_index_cache = {}

    key = ("chain" if use_chain_edges else "fc", numFPs, device.type)
    if key not in edge_index_cache:
        if use_chain_edges:
            edge_index_cache[key] = _chain_edge_index(numFPs, device=device, bidirectional=True)
        else:
            edge_index_cache[key] = _fully_connected_edge_index(numFPs, device=device, self_loops=False)
    base_ei = edge_index_cache[key]                                     # [2,E]
    E = base_ei.size(1)

    # replicate edges across batch with offsets
    offsets = (torch.arange(B, device=device) * numFPs).view(B, 1, 1)    # [B,1,1]
    ei = base_ei.view(1, 2, E) + offsets                                 # [B,2,E]
    edge_index = ei.permute(1, 0, 2).reshape(2, B * E)                   # [2, B*E]

    # ---------------- Edge features (vectorized) ----------------
    src = base_ei[0]   # [E]
    dst = base_ei[1]   # [E]

    node_xyz = fps_pos                                                   # [B,N,3]
    delta = node_xyz[:, dst, :] - node_xyz[:, src, :]                    # [B,E,3]
    dist = torch.norm(delta, dim=-1, keepdim=True) + eps                 # [B,E,1]
    unit_dir = delta / dist                                              # [B,E,3]

    if use_length_norm:
        dist_feat = dist / length_safe.view(B, 1, 1)                     # [B,E,1]
    else:
        dist_feat = dist

    if add_distance_to_edge:
        edge_attr_base = torch.cat([unit_dir, dist_feat], dim=-1)        # [B,E,4]
    else:
        edge_attr_base = unit_dir                                        # [B,E,3]

    # ---------- NEW: concat node features i, node features j, and base edge attr ----------
    # x is [B,N,x_dim] (BEFORE flatten)
    x_src = x[:, src, :]                                                 # [B,E,x_dim]
    x_dst = x[:, dst, :]                                                 # [B,E,x_dim]

    edge_attr = torch.cat([x_src, x_dst, edge_attr_base], dim=-1)        # [B,E, 2*x_dim + (3/4)]
    edge_attr = edge_attr.reshape(B * E, -1)                             # [B*E, edge_dim_new]

    # -----------------------------------------------
    # NEW: mask edges connected to missing nodes
    valid_edge = present_mask[:, src] & present_mask[:, dst]   # [B, E] bool

    # flatten everything and filter
    valid_edge_flat = valid_edge.reshape(B * E)                # [B*E]

    edge_index = edge_index[:, valid_edge_flat]                # [2, E_valid_total]
    edge_attr  = edge_attr.reshape(B * E, -1)[valid_edge_flat] # [E_valid_total, edge_dim]

    # (optional but recommended) also zero-out missing node features
    # so isolated nodes don’t inject garbage if your model reads x anyway
    x = x.reshape(B, numFPs, -1)
    x = x * present_mask.unsqueeze(-1)                         # [B,N,xdim]
    x_flat = x.reshape(B * numFPs, -1)
    # -------------------------------------------------

    # batch vector for nodes
    batch = torch.arange(B, device=device).repeat_interleave(numFPs)     # [B*N]

    data = pyg.data.Data(
        x=x_flat,
        edge_index=edge_index,
        edge_attr=edge_attr,
        u=u,          # keep graph-level u if your model needs it
        batch=batch,
    )
    return data, edge_index_cache
