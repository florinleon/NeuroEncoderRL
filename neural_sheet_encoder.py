import math
import random
from typing import List, Sequence, Optional
import numpy as np
from settings import *


try:
    from numba import njit
except Exception:
    njit = None


def _ensure_1d_float_array(x):
    # Utility: convert any 1D sequence to a float32 NumPy array without changing the shape.
    arr = np.asarray(x, dtype=np.float32)
    return arr


if njit is not None:

    @njit
    def _lateral_inhibition_dynamics_jit(u, kernel, inhibition_strength, leak_rate, time_steps):
        """
        Lateral inhibition dynamics on a toroidal sheet.

        u: feedforward drive, shape (H, W)
        kernel: inhibition kernel, shape (K, K), K = 2*r+1, center weight must be 0
        """
        height, width = u.shape
        ksize = kernel.shape[0]
        r = ksize // 2

        v = u.copy()

        for _ in range(time_steps):
            v_new = np.empty_like(v)
            for y in range(height):
                for x in range(width):
                    inhibition = 0.0
                    for dy in range(-r, r + 1):
                        ky = dy + r
                        for dx in range(-r, r + 1):
                            kx = dx + r
                            if dx == 0 and dy == 0:
                                continue

                            ny = (y + dy + height) % height
                            nx = (x + dx + width) % width

                            w = kernel[ky, kx]
                            if w != 0.0:
                                inhibition += w * v[ny, nx]

                    current = v[y, x]
                    drive = u[y, x] - current - inhibition_strength * inhibition
                    updated = current + leak_rate * drive
                    if updated < 0.0:
                        updated = 0.0

                    v_new[y, x] = updated
            v = v_new

        return v

else:
    _lateral_inhibition_dynamics_jit = None


class NeuralSheetEncoder:
    """
    Continuous-to-sheet encoder with lateral inhibition and sparsification.

    The encoder maps an input vector in [0,1]^D into a 2D neural sheet.
    It uses Gaussian population codes per input dimension, a sparse random
    projection from detectors to sheet units, and recurrent lateral inhibition
    followed by percentile-based sparsification. Similarity between codes is
    a graded overlap on the sheet.
    """

    def __init__(self, random_seed = None,
                 dimension_gains = None):
        # Basic geometry
        self._input_dimension_count = InputDimensionCount
        self._detectors_per_dimension = DetectorsPerDimension
        self._total_detectors = self._input_dimension_count * self._detectors_per_dimension
        self._sheet_width = SheetWidth
        self._sheet_height = SheetHeight
        self._sheet_size = self._sheet_width * self._sheet_height

        # Per-dimension detector gains (input-axis weighting).
        if dimension_gains is None:
            self._dimension_gains = np.ones(self._input_dimension_count, dtype=np.float32)
        else:
            gains_arr = _ensure_1d_float_array(dimension_gains)
            self._dimension_gains = gains_arr

        # Random generator
        self._random = random.Random(random_seed)

        # Detector centers as a (D, K) numpy array
        self._detector_centers = self._create_detector_centers_np()

        # Random projection matrix as numpy array (sheet_size, total_detectors)
        self._random_projection = self._create_random_projection_matrix_np(
            self._sheet_size, self._total_detectors
        )

        # Inhibition kernel as numpy array
        self._inhibition_radius = int(math.ceil(3.0 * InhibitionSigma))
        self._inhibition_kernel = self._create_inhibition_kernel_np(self._inhibition_radius)


    def encode(self, input_vec):
        """
        Encode a continuous input vector (values in [0,1]) into a 2D sheet.

        Returns a list-of-lists [SheetHeight][SheetWidth] with activations
        after lateral inhibition and sparsification. The sheet size and all
        dynamical parameters are taken from the constants defined in settings.py.
        """
        # 1) Gaussian population code per dimension
        x = _ensure_1d_float_array(input_vec)
        x_clipped = np.clip(x, 0.0, 1.0)
        detectors = self._encode_to_detectors_np(x_clipped)

        # 2) Sparse random projection to sheet (pre-inhibition)
        u_flat = self._random_projection @ detectors  # shape: (sheet_size,)
        u = u_flat.reshape(self._sheet_height, self._sheet_width)

        # 3) Recurrent lateral inhibition dynamics on the sheet
        if _lateral_inhibition_dynamics_jit is not None:
            v = _lateral_inhibition_dynamics_jit(u, self._inhibition_kernel, float(InhibitionStrength), float(LeakRate), int(TimeSteps))
        else:
            # Fallback to pure Python version if Numba is not available
            v_list = self._lateral_inhibition_loop(u.tolist(), u.tolist())
            v = np.asarray(v_list, dtype=np.float32)

        
        # Optional small neural noise before sparsification
        if UseNoise and NoiseStd > 0.0:
            seed = self._random.randint(0, 2**31 - 1)
            rng = np.random.default_rng(seed)
            noise = rng.normal(loc=0.0, scale=NoiseStd, size=v.shape).astype(np.float32)
            v = v + noise
            np.maximum(v, 0.0, out=v)

        # 4) Global sparsification via fixed top-k (derived from SparsityPercentile)
        flat = v.ravel()
        if flat.size == 0:
            result = v
        else:
            p = SparsityPercentile
            if p <= 0.0:
                # No sparsification: keep all units
                result = v
            else:
                n = flat.size
                fraction_keep = 1.0 - p
                if fraction_keep <= 0.0:
                    k = 1
                elif fraction_keep >= 1.0:
                    k = n
                else:
                    k = int(math.floor(fraction_keep * n))
                    if k < 1:
                        k = 1

                flat_copy = flat.copy()
                # Indices of the k largest entries
                topk_idx = np.argpartition(flat_copy, -k)[-k:]
                mask = np.zeros_like(flat_copy, dtype=bool)
                mask[topk_idx] = True

                flat_sparse = np.zeros_like(flat_copy)
                flat_sparse[mask] = flat_copy[mask]
                result = flat_sparse.reshape(v.shape)

                return result.tolist()


    def set_dimension_gains(self, gains):
            """Set per-input-dimension gains for detector activations.

            The gains sequence must have length equal to the input dimensionality (D).
            """
            arr = _ensure_1d_float_array(gains)
            self._dimension_gains = arr


    def similarity(self, a, b):
        """
        Graded assembly-overlap similarity between two 2D activation maps.

        Both a and b are expected to be [height][width] lists produced by
        this encoder. Each map is first normalized by its own maximum value
        (if positive), and similarity is computed over the union of sites
        that are active in at least one map.
        """
        a_np = np.asarray(a, dtype=np.float32)
        b_np = np.asarray(b, dtype=np.float32)

        height = min(a_np.shape[0], b_np.shape[0])
        width = min(a_np.shape[1], b_np.shape[1])
        if height == 0 or width == 0:
            return 0.0

        a_sub = a_np[:height, :width]
        b_sub = b_np[:height, :width]

        a_max = float(a_sub.max())
        if a_max > 0.0:
            a_norm = a_sub / a_max
        else:
            a_norm = np.zeros_like(a_sub)

        b_max = float(b_sub.max())
        if b_max > 0.0:
            b_norm = b_sub / b_max
        else:
            b_norm = np.zeros_like(b_sub)

        mask = ~((a_norm == 0.0) & (b_norm == 0.0))
        if not mask.any():
            return 0.0

        local_sim = 1.0 - np.abs(a_norm - b_norm)
        np.maximum(local_sim, 0.0, out=local_sim)

        sum_local = float(local_sim[mask].sum())
        active_count = int(mask.sum())
        return sum_local / float(active_count)


    def flatten(self, activations):
        """
        Flatten a 2D activation map [height][width] into a 1D list.
        """
        arr = np.asarray(activations, dtype=np.float32)
        return arr.ravel().tolist()


    def pretty_print_activations(self, input_vec):
        """
        Encode the input and print the activation map using two decimals.
        """
        activations = self.encode(input_vec)
        arr = np.asarray(activations, dtype=np.float32)

        height, width = arr.shape
        if height == 0 or width == 0:
            return

        for y in range(height):
            row_str = "".join(f"{arr[y, x]:6.2f}" for x in range(width))
            print(row_str)


    # ------------------------------------------------------------------
    # Internal helpers (NumPy-based)
    # ------------------------------------------------------------------

    def _create_detector_centers_np(self):
        """
        Create detector centers per dimension as a (D, K) numpy array.

        Centers for each dimension lie in an extended range [-0.1, 1.1],
        evenly spaced, so that boundary inputs in [0,1] have neighbors on
        both sides in detector space.
        """
        d = self._input_dimension_count
        k = self._detectors_per_dimension

        if k <= 1:
            centers_1d = np.array([0.5], dtype=np.float32)
        else:
            centers_1d = np.linspace(-0.1, 1.1, num=k, dtype=np.float32)

        centers = np.empty((d, k), dtype=np.float32)
        for dim in range(d):
            centers[dim, :] = centers_1d

        return centers


    def _create_random_projection_matrix_np(self, sheet_size, detector_count):
        """
        Create a sparse random projection matrix as a dense NumPy array
        with L2-normalized rows and zero-mean signed weights.
        """
        fan_in = FanIn
        if fan_in > detector_count:
            fan_in = detector_count

        mat = np.zeros((sheet_size, detector_count), dtype=np.float32)

        for i in range(sheet_size):
            # Choose fan_in distinct detector indices for this sheet unit
            chosen = set()
            while len(chosen) < fan_in:
                idx = self._random.randint(0, detector_count - 1)
                if idx not in chosen:
                    chosen.add(idx)

            idx_list = list(chosen)

            # Draw signed weights for these connections
            weights = np.array([self._random.gauss(0.0, 1.0) for _ in range(fan_in)], dtype=np.float32)

            # Enforce zero mean over the nonzero entries
            mean_w = float(weights.mean())
            weights = weights - mean_w

            # L2-normalize the row over the nonzero entries
            norm = math.sqrt(float((weights ** 2).sum()))
            if norm > 0.0:
                weights /= norm

            # Write the nonzero weights into the row
            mat[i, idx_list] = weights

        return mat


    def _create_inhibition_kernel_np(self, radius):
        """
        Create a Gaussian inhibition kernel as a NumPy array of shape (K, K),
        with zero at the center position.
        """
        sigma = InhibitionSigma
        size = 2 * radius + 1
        kernel = np.zeros((size, size), dtype=np.float32)

        total = 0.0
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                y = dy + radius
                x = dx + radius

                if dx == 0 and dy == 0:
                    kernel[y, x] = 0.0
                    continue

                distance_sq = float(dx * dx + dy * dy)
                value = math.exp(-distance_sq / (2.0 * sigma * sigma))
                kernel[y, x] = value
                total += value

        if total > 0.0:
            kernel /= float(total)

        return kernel


    def _encode_to_detectors_np(self, x):
        """Encode the input vector into a flattened detector activity vector."""
        sigma = GaussianSigma
        diffs = x[:, None] - self._detector_centers  # (D, K)
        detectors = np.exp(-0.5 * (diffs * diffs) / (sigma * sigma)).astype(np.float32)
        if getattr(self, "_dimension_gains", None) is not None:
            detectors *= self._dimension_gains[:, None]
        return detectors.reshape(self._total_detectors)


    # ------------------------------------------------------------------
    # Fallback pure-Python dynamics (used only if Numba is not available)
    # ------------------------------------------------------------------

    def _lateral_inhibition_loop(self, v_list, u_list):
        """
        Pure Python lateral inhibition loop, used only if Numba is not available.
        v_list and u_list are list-of-lists.
        """
        height = len(v_list)
        width = len(v_list[0])
        r = self._inhibition_radius
        kernel = self._inhibition_kernel.tolist()

        v = [row[:] for row in v_list]
        u = [row[:] for row in u_list]

        for _ in range(TimeSteps):
            next_v = [[0.0 for _ in range(width)] for _ in range(height)]
            for y in range(height):
                for x in range(width):
                    inhibition = 0.0
                    for dy in range(-r, r + 1):
                        for dx in range(-r, r + 1):
                            if dx == 0 and dy == 0:
                                continue
                            ny = (y + dy + height) % height
                            nx = (x + dx + width) % width
                            weight = kernel[dy + r][dx + r]
                            inhibition += weight * v[ny][nx]

                    current = v[y][x]
                    drive = u[y][x] - current - InhibitionStrength * inhibition
                    updated = current + LeakRate * drive
                    if updated < 0.0:
                        updated = 0.0
                    next_v[y][x] = updated

            v = next_v

        return v


    # ------------------------------------------------------------------
    # For kWTA
    # ------------------------------------------------------------------

    def _compute_percentile_np(self, values):
        """
        Compute the global percentile threshold.
        """
        flat = values.ravel()
        if flat.size == 0:
            return 0.0

        p = SparsityPercentile
        if p <= 0.0:
            return float(flat.min())
        if p >= 1.0:
            return float(flat.max())

        idx = int(math.floor(p * (flat.size - 1)))
        flat_copy = flat.copy()
        threshold = float(np.partition(flat_copy, idx)[idx])
        return threshold
