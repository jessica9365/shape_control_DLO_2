#!/usr/bin/env python3
import os
import numpy as np
import argparse

from utils.state_index import I  # must match your project indices


parser = argparse.ArgumentParser("Preprocess dataset and generate mask variants")
parser.add_argument(
    "--projectdir",
    type=str,
    default="/home/jessica/shape_control_DLO_2/",
)
parser.add_argument(
    "--envdim",
    type=str,
    default="2D",  # used in path: data/train_data/<envdim>
)
parser.add_argument(
    "--numFPs",
    type=int,
    default=10,
    help="Number of feature points on the DLO",
)
parser.add_argument(
    "--start_id",
    type=int,
    default=1,
    help="First state_X.npy to process (inclusive)",
)
parser.add_argument(
    "--end_id",
    type=int,
    default=10,
    help="Last state_X.npy to process (inclusive)",
)
parser.add_argument(
    "--seed",
    type=int,
    default=0,
    help="Random seed for occlusion patterns",
)

args = parser.parse_args()


def load_state(path):
    data = np.load(path).astype(np.float32)
    if data.ndim != 2:
        raise ValueError(f"{path} has shape {data.shape}, expected 2D [T, D]")
    return data


def save_with_mask(base_dir, subdir, sid, data, mask_suffix):
    """
    base_dir: e.g. <projectdir>/data/train_data/2D
    subdir: e.g. '2D_mask_1'
    sid: integer state id
    data: [T, D] or [T, D + numFPs]
    mask_suffix: e.g. 'mask_1', 'mask_variable', 'mixed'
    """
    out_dir = os.path.join(base_dir, subdir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"state_{sid}_{mask_suffix}.npy")
    np.save(out_path, data)
    print(f"Saved: {out_path} shape={data.shape}")


def append_all_ones_mask(data, numFPs):
    """
    data: [T, D]
    returns [T, D + numFPs] with mask=1 for all timesteps and FPs
    """
    T, D = data.shape
    out = np.zeros((T, D + numFPs), dtype=np.float32)
    out[:, :D] = data
    out[:, D:] = 1.0
    return out


def sample_contiguous_indices(rng, numFPs, k):
    """
    Sample k contiguous FP indices in [0, numFPs-1].
    """
    if k >= numFPs:
        return list(range(numFPs))
    start_max = numFPs - k
    start = rng.integers(0, start_max + 1)
    return list(range(start, start + k))


def sample_noncontiguous_indices(rng, numFPs, k):
    """
    Sample k distinct, non-contiguous FP indices in [0, numFPs-1].
    We only enforce 'distinct'; contiguity is unlikely for small k.
    """
    if k >= numFPs:
        return list(range(numFPs))
    return list(rng.choice(numFPs, size=k, replace=False))


def build_random_mask_array(rng, T, numFPs):
    """
    For each timestep independently, construct a mask row:
      - With prob 0.5: contiguous occlusion
      - With prob 0.5: non-contiguous occlusion
      - Occlude 1–4 FPs (if numFPs >= 4, otherwise up to numFPs)
      - mask[i] = 1 (visible), 0 (occluded)
    Returns mask: [T, numFPs]
    """
    mask = np.ones((T, numFPs), dtype=np.float32)
    max_occlude = min(4, numFPs)

    for t in range(T):
        # decide number of occluded FPs
        k = int(rng.integers(1, max_occlude + 1))

        # contiguous vs non-contiguous
        if rng.random() < 0.5:
            occluded = sample_contiguous_indices(rng, numFPs, k)
        else:
            occluded = sample_noncontiguous_indices(rng, numFPs, k)

        mask[t, occluded] = 0.0

    return mask


def create_mask_variable_version(rng, data, numFPs):
    """
    data: [T, D] original state
    returns [T, D + numFPs]: original data + random mask (0/1)
    """
    T, D = data.shape
    out = np.zeros((T, D + numFPs), dtype=np.float32)
    out[:, :D] = data
    mask = build_random_mask_array(rng, T, numFPs)
    out[:, D:] = mask
    return out


def create_mixed_version(rng, data, numFPs):
    """
    data: [T, D]
    returns [T, D + numFPs] with a mix of:
      - some timesteps no occlusion (all ones)
      - some timesteps occlusion (0/1)
    Strategy:
      - For each timestep:
          with prob 1/3: no occlusion (all ones)
          with prob 2/3: occlusion with same scheme as mask_variable
    """
    T, D = data.shape
    out = np.zeros((T, D + numFPs), dtype=np.float32)
    out[:, :D] = data

    mask = np.ones((T, numFPs), dtype=np.float32)
    max_occlude = min(4, numFPs)

    for t in range(T):
        r = rng.random()
        if r < 1.0 / 3.0:
            # no occlusion, keep all 1's
            continue
        else:
            # occlusion case
            k = int(rng.integers(1, max_occlude + 1))
            if rng.random() < 0.5:
                occluded = sample_contiguous_indices(rng, numFPs, k)
            else:
                occluded = sample_noncontiguous_indices(rng, numFPs, k)
            mask[t, occluded] = 0.0

    out[:, D:] = mask
    return out


def main():
    rng = np.random.default_rng(args.seed)

    base_dir = os.path.join(args.projectdir, "data", "train_data", args.envdim)
    assert os.path.isdir(base_dir), f"Data dir not found: {base_dir}"

    for sid in range(args.start_id, args.end_id + 1):
        in_path = os.path.join(base_dir, f"state_{sid}.npy")
        if not os.path.isfile(in_path):
            print(f"WARNING: {in_path} not found, skipping.")
            continue

        data = load_state(in_path)

        # 1) 2D_mask_1: all ones mask
        data_mask_1 = append_all_ones_mask(data, args.numFPs)
        save_with_mask(
            base_dir,
            subdir="2D_mask_1",
            sid=sid,
            data=data_mask_1,
            mask_suffix="mask_1",
        )

        # 2) 2D_mask_variable: random occlusion every timestep
        data_mask_var = create_mask_variable_version(rng, data, args.numFPs)
        save_with_mask(
            base_dir,
            subdir="2D_mask_variable",
            sid=sid,
            data=data_mask_var,
            mask_suffix="mask_variable",
        )

        # 3) 2D_mixed: some timesteps full visibility, some occluded
        data_mixed = create_mixed_version(rng, data, args.numFPs)
        save_with_mask(
            base_dir,
            subdir="2D_mixed",
            sid=sid,
            data=data_mixed,
            mask_suffix="mixed",
        )


if __name__ == "__main__":
    main()
