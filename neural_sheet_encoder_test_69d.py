"""Neural sheet similarity check for 69-D Bridge-style inputs (similarity-only).

It uses 69-D inputs with the same structure as the bridge RL observation encoding:
  - 9 local_grid cells, each encoded as a 7-way one-hot (63 dims)
  - 2 continuous position dims in [0,1]
  - time one-hot (2 dims)
  - carry one-hot (2 dims)
"""

import random
from typing import List, Tuple, Dict, Any
import numpy as np
from neural_sheet_encoder import NeuralSheetEncoder
from settings import InputDimensionCount


NumCellClasses = 7
NumLocalCells = 9


def _one_hot(idx, k):
    # Simple k-way one-hot encoder used for the discrete local_grid and flag bits.
    v = np.zeros(k, dtype=np.float32)
    if 0 <= idx < k:
        v[idx] = 1.0
    return v


def make_bridge_like_x(cell_classes, pos, time_indicator, has_object):
    """
    Pack a Bridge-style observation into a 69-D vector.

    The layout is:
      - 9 local_grid entries, each mapped to a 7-way one-hot (63 dims total)
      - 2 normalized position coordinates (row, col) in [0,1]
      - 2 one-hot bits for time-of-day
      - 2 one-hot bits for object possession
    """
    local_oh = np.concatenate([_one_hot(int(c), NumCellClasses) for c in cell_classes], dtype=np.float32)

    row_norm = float(pos[0])
    col_norm = float(pos[1])
    row_norm = max(0.0, min(1.0, row_norm))
    col_norm = max(0.0, min(1.0, col_norm))
    pos_vec = np.array([row_norm, col_norm], dtype=np.float32)

    t = int(time_indicator)
    time_oh = np.array([1.0, 0.0], dtype=np.float32) if t == 0 else np.array([0.0, 1.0], dtype=np.float32)

    c = int(has_object)
    carry_oh = np.array([1.0, 0.0], dtype=np.float32) if c == 0 else np.array([0.0, 1.0], dtype=np.float32)

    x = np.concatenate([local_oh, pos_vec, time_oh, carry_oh], dtype=np.float32)
    x = np.clip(x, 0.0, 1.0)

    return x


def semantic_cell_diffs(a_cells, b_cells):
    return sum(int(x != y) for x, y in zip(a_cells, b_cells))


def build_point_suite(rng):
    """
    Construct a fixed dictionary of labeled perturbation cases.

    Each entry describes a small change (continuous, discrete, or both) relative
    to a shared reference point; this allows similarity scores to be compared
    against simple, human-readable differences.
    """
    Floor = 1
    Water = 2
    Bridge = 4

    base_cells = [Floor] * NumLocalCells
    base_pos = (0.40, 0.40)
    base_time = 0
    base_carry = 0

    points: Dict[str, Dict[str, Any]] = {}
    points["reference"] = dict(cells=base_cells, pos=base_pos, time=base_time, carry=base_carry)

    # Continuous position changes (same discrete bits)
    points["pos_small_move"] = dict(cells=base_cells, pos=(base_pos[0] + 0.10, base_pos[1]), time=base_time, carry=base_carry)
    points["pos_large_move"] = dict(cells=base_cells, pos=(base_pos[0] + 0.50, base_pos[1]), time=base_time, carry=base_carry)

    # Discrete flips
    points["time_flip_only"] = dict(cells=base_cells, pos=base_pos, time=1, carry=base_carry)
    points["carry_flip_only"] = dict(cells=base_cells, pos=base_pos, time=base_time, carry=1)
    points["time_and_carry_flip"] = dict(cells=base_cells, pos=base_pos, time=1, carry=1)

    # Local-grid edits (one-hot index changes)
    one_cell_water = base_cells.copy()
    one_cell_water[4] = Water
    points["one_cell_water"] = dict(cells=one_cell_water, pos=base_pos, time=base_time, carry=base_carry)

    three_cells_water = base_cells.copy()
    for idx in [0, 4, 8]:
        three_cells_water[idx] = Water
    points["three_cells_water"] = dict(cells=three_cells_water, pos=base_pos, time=base_time, carry=base_carry)

    one_cell_bridge = base_cells.copy()
    one_cell_bridge[4] = Bridge
    points["one_cell_bridge"] = dict(cells=one_cell_bridge, pos=base_pos, time=base_time, carry=base_carry)

    # Random neighborhoods ("far" in discrete part)
    def random_cells():
        return [rng.randint(0, 6) for _ in range(NumLocalCells)]
    points["random_cells_1"] = dict(cells=random_cells(), pos=base_pos, time=base_time, carry=base_carry)
    points["random_cells_2"] = dict(cells=random_cells(), pos=(0.90, 0.10), time=1, carry=1)

    return points


def position_axis_test(runs, steps = 10, master_seed = 0):
    """
    Measure how neural-sheet similarity decays as the row coordinate moves
    along a single spatial axis while all other input components stay fixed.
    """
    global_rng = random.Random(master_seed)

    Floor = 1
    base_cells = [Floor] * NumLocalCells
    base_time = 0
    base_carry = 0
    col_fixed = 0.40

    # reference at row=0
    ref_x = make_bridge_like_x(base_cells, (0.0, col_fixed), base_time, base_carry)

    sum_sims = [0.0] * (steps + 1)

    for _ in range(runs):
        encoder = NeuralSheetEncoder(random_seed=global_rng.randint(0, 2**31 - 1))
        ref_code = encoder.encode(ref_x)

        for i in range(steps + 1):
            row = i / steps
            x = make_bridge_like_x(base_cells, (row, col_fixed), base_time, base_carry)
            code = encoder.encode(x)
            sum_sims[i] += encoder.similarity(ref_code, code)

    print(f"[Test 1] Continuous row change (col fixed). Avg over {runs} encoder seeds")
    print("row_norm\tAvgSimilarity")
    for i in range(steps + 1):
        row = i / steps
        print(f"{row:.2f}\t\t{(sum_sims[i]/runs):.3f}")
    print()


def mixed_point_suite(runs, master_seed = 1):
    global_rng = random.Random(master_seed)

    # Fix the point suite; average only over encoder randomness
    point_rng = random.Random(12345)
    points = build_point_suite(point_rng)

    ref = points["reference"]
    ref_x = make_bridge_like_x(ref["cells"], ref["pos"], ref["time"], ref["carry"])

    names = [k for k in points.keys() if k != "reference"]
    sum_sims = {name: 0.0 for name in names}

    # Pre-compute simple, interpretable differences (not norms)
    diffs = {}
    for name in names:
        p = points[name]
        diffs[name] = dict(
            changed_cells=semantic_cell_diffs(ref["cells"], p["cells"]),
            pos_delta=((p["pos"][0] - ref["pos"][0])**2 + (p["pos"][1] - ref["pos"][1])**2) ** 0.5,
            time_flip=int(p["time"] != ref["time"]),
            carry_flip=int(p["carry"] != ref["carry"]),
        )

    for _ in range(runs):
        encoder = NeuralSheetEncoder(random_seed=global_rng.randint(0, 2**31 - 1))
        ref_code = encoder.encode(ref_x)

        for name in names:
            p = points[name]
            x = make_bridge_like_x(p["cells"], p["pos"], p["time"], p["carry"])
            code = encoder.encode(x)
            sum_sims[name] += encoder.similarity(ref_code, code)

    print(f"[Test 2] Mixed perturbations. Avg over {runs} encoder seeds")
    print("Case\t\t\tChangedCells\tPosDelta\tTimeFlip\tCarryFlip\tAvgSimilarity")
    for name in names:
        d = diffs[name]
        avg_sim = sum_sims[name] / runs
        print(f"{name:18s}\t{d['changed_cells']:11d}\t{d['pos_delta']:7.3f}\t{d['time_flip']:8d}\t{d['carry_flip']:9d}\t{avg_sim:11.3f}")
    print()


def similarity_vs_changed_cells(runs, trials_per_k = 50, master_seed = 2):
    global_rng = random.Random(master_seed)
    perturb_rng = random.Random(2026)

    Floor = 1
    base_cells = [Floor] * NumLocalCells
    base_pos = (0.40, 0.40)
    base_time = 0
    base_carry = 0

    ref_x = make_bridge_like_x(base_cells, base_pos, base_time, base_carry)

    # Pre-sample perturbations for each k (0..9)
    perturbations = {k: [] for k in range(NumLocalCells + 1)}
    for k in range(NumLocalCells + 1):
        for _ in range(trials_per_k):
            cells = base_cells.copy()
            idxs = perturb_rng.sample(range(NumLocalCells), k)
            for idx in idxs:
                new_class = perturb_rng.randint(0, 6)
                if new_class == cells[idx]:
                    new_class = (new_class + 1) % 7
                cells[idx] = new_class
            perturbations[k].append(cells)

    sum_sims = {k: 0.0 for k in range(NumLocalCells + 1)}

    for _ in range(runs):
        encoder = NeuralSheetEncoder(random_seed=global_rng.randint(0, 2**31 - 1))
        ref_code = encoder.encode(ref_x)

        for k in range(NumLocalCells + 1):
            for cells in perturbations[k]:
                x = make_bridge_like_x(cells, base_pos, base_time, base_carry)
                code = encoder.encode(x)
                sum_sims[k] += encoder.similarity(ref_code, code)

    denom = float(runs * trials_per_k)
    print(f"[Test 3] Similarity vs number of changed local_grid cells. Avg over {runs} encoder seeds, {trials_per_k} trials/k")
    print("ChangedCells\tAvgSimilarity")
    for k in range(NumLocalCells + 1):
        print(f"{k:11d}\t{(sum_sims[k]/denom):.3f}")
    print()


if __name__ == "__main__":
    runs = 100
    position_axis_test(runs=runs, steps=10, master_seed=0)
    mixed_point_suite(runs=runs, master_seed=1)
    similarity_vs_changed_cells(runs=runs, trials_per_k=10, master_seed=2)
