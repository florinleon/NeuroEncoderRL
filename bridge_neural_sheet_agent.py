import numpy as np
from typing import Optional, Dict, Any
from neural_sheet_encoder import NeuralSheetEncoder


# ----------------------------------------------------------------------
# Observation encoding for the bridge agents
# ----------------------------------------------------------------------

def one_hot_cell(cell_value, num_classes = 7):
    """One-hot encode a local_grid cell value (0..6)."""
    # Maps a raw BridgeEnv observation into the fixed 69-D feature vector.
    oh = np.zeros(num_classes, dtype=np.float32)
    iv = int(cell_value)
    if 0 <= iv < num_classes:
        oh[iv] = 1.0
    return oh


def flatten_obs(obs):
    """Construct the fixed 69-D observation vector used by the bridge agents.

    Layout:
      - 9 local_grid entries × 7 one-hot  -> 63
      - agent_pos (row, col)             -> 2
      - time_indicator one-hot           -> 2
      - has_object one-hot               -> 2
    """
    local = np.asarray(obs["local_grid"], dtype=np.int32).reshape(-1)
    local_oh = np.concatenate([one_hot_cell(int(v)) for v in local], dtype=np.float32)

    pos = np.asarray(obs["agent_pos"], dtype=np.float32).reshape(-1)

    t = int(obs["time_indicator"])
    time_oh = np.array([1.0, 0.0], dtype=np.float32) if t == 0 else np.array([0.0, 1.0], dtype=np.float32)

    carry = int(obs["has_object"])
    carry_oh = np.array([1.0, 0.0], dtype=np.float32) if carry == 0 else np.array([0.0, 1.0], dtype=np.float32)

    return np.concatenate([local_oh, pos, time_oh, carry_oh], dtype=np.float32)


def bridge_obs_to_input(obs, grid_min = 1.0, grid_max = 6.0):
    """Map BridgeEnv observation to the 69-D continuous input in [0,1]^69.

    Important: This preserves the same 69-D layout as ``flatten_obs``; the only
    additional step is normalization of the 2 position coordinates to [0,1].
    """
    v = flatten_obs(obs).astype(np.float32).copy()

    # Positions are entries [63, 64] in the 69-D layout.
    local_len = 9 * 7
    pos_start = local_len
    pos_end = pos_start + 2

    denom = max(1.0, float(grid_max - grid_min))
    v[pos_start:pos_end] = (v[pos_start:pos_end] - grid_min) / denom

    return np.clip(v, 0.0, 1.0)


# ----------------------------------------------------------------------
# RL agent: linear value approximation on top of neural-sheet features
# ----------------------------------------------------------------------

class NeuralSheetQAgent:
    """Monte-Carlo / TD-capable linear agent over neural-sheet features.

    Representation:
      - x(s): 69-D vector from bridge_obs_to_input.
      - R(x): (H, W) neural sheet activity map (sparse).
      - phi(s): flattened R(x), optionally L2-normalized.

    Value function:
      Q(s, a) = w[a] dot phi(s)

    Notes:
      - The code includes both 1-step Q-learning update and a Monte-Carlo update.
      - The training script uses Monte-Carlo (first-visit) updates because the task is sparse-reward and POMDP-like.
    """

    def __init__(
        self,
        encoder,
        num_actions,
        learning_rate = 0.01,
        discount = 0.99,
        epsilon_start = 1.0,
        epsilon_end = 0.05,
        epsilon_decay_steps = 500_000,
        seed = None,
        normalize_features = True,
        td_clip = 10.0,
        weight_clip = 1000.0,
    ):
        self.encoder = encoder
        self.num_actions = int(num_actions)
        self.learning_rate = float(learning_rate)
        self.discount = float(discount)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay_steps = int(max(1, epsilon_decay_steps))
        self.normalize_features = bool(normalize_features)

        self.td_clip = td_clip
        self.weight_clip = weight_clip

        self.rng = np.random.RandomState(seed)
        self.steps_done = 0

        # Determine feature dimensionality from a dummy encoding
        dummy_x = np.full(self.encoder._input_dimension_count, 0.5, dtype=np.float32)
        dummy_sheet = np.asarray(self.encoder.encode(dummy_x), dtype=np.float32)
        self.sheet_height, self.sheet_width = dummy_sheet.shape
        self.feature_dim = self.sheet_height * self.sheet_width

        self.weights = np.zeros((self.num_actions, self.feature_dim), dtype=np.float32)


    # ------------------------------------------------------------------
    # Features
    # ------------------------------------------------------------------
    
    def _phi_from_input(self, x):
        sheet = np.asarray(self.encoder.encode(x), dtype=np.float32)
        phi = sheet.reshape(-1)
        if self.normalize_features:
            n = float(np.linalg.norm(phi))
            if n > 0.0:
                phi = phi / n
        return phi


    def features_from_obs(self, obs):
        x = bridge_obs_to_input(obs)
        return self._phi_from_input(x)


    # ------------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------------
    
    def epsilon(self):
        frac = min(1.0, self.steps_done / float(self.epsilon_decay_steps))
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)


    def select_action(self, obs, greedy = False):
        """Epsilon-greedy action selection.

        Args:
            obs: environment observation dict
            greedy: if True, select argmax action without exploration

        Returns:
            (action, phi) where phi is the neural-sheet feature vector for obs.
        """
        phi = self.features_from_obs(obs)

        q_values = self.weights.dot(phi)  # shape (A,)

        eps = 0.0 if greedy else self.epsilon()
        self.steps_done += 1

        if (not greedy) and (self.rng.rand() < eps):
            action = int(self.rng.randint(self.num_actions))
            return action, phi

        action = int(np.argmax(q_values))
        return action, phi


    # ------------------------------------------------------------------
    # Learning updates
    # ------------------------------------------------------------------

    def _clip_scalar(self, value):
        if self.td_clip is None:
            return float(value)
        if value > self.td_clip:
            return float(self.td_clip)
        if value < -self.td_clip:
            return float(-self.td_clip)
        return float(value)


    def update_mc(self, phi, action, target_return):
        """Monte-Carlo regression step toward a return target G.

        w[a] <- w[a] + lr * (G - Q(s,a)) * phi
        """
        q_a = float(self.weights[action, :].dot(phi))
        err = float(target_return) - q_a
        err = self._clip_scalar(err)

        self.weights[action, :] += self.learning_rate * err * phi

        if self.weight_clip is not None:
            np.clip(self.weights, -self.weight_clip, self.weight_clip, out=self.weights)

        return err


    def update_qlearning(self, phi, action, reward, next_obs, done):
        """One-step Q-learning update."""
        q_current = self.weights.dot(phi)
        q_a = float(q_current[action])

        if next_obs is None or done:
            target = float(reward)
        else:
            next_phi = self.features_from_obs(next_obs)
            q_next = self.weights.dot(next_phi)
            target = float(reward) + self.discount * float(np.max(q_next))

        td_error = self._clip_scalar(target - q_a)

        self.weights[action, :] += self.learning_rate * td_error * phi

        if self.weight_clip is not None:
            np.clip(self.weights, -self.weight_clip, self.weight_clip, out=self.weights)

        return float(td_error)


    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self, path):
        """Save full agent state (weights, exploration progress, RNG, encoder)."""
        rng_state = self.rng.get_state()
        save_dict = {
            "weights": self.weights,
            "steps_done": np.int64(self.steps_done),
            # Store RNG state as an object array to avoid shape issues.
            "rng_state": np.array(rng_state, dtype=object),
        }

        # Serialize encoder parameters as well (neural sheet state)
        enc = getattr(self, "encoder", None)
        if enc is not None:
            if hasattr(enc, "_dimension_gains"):
                save_dict["encoder_dimension_gains"] = enc._dimension_gains
            if hasattr(enc, "_detector_centers"):
                save_dict["encoder_detector_centers"] = enc._detector_centers
            if hasattr(enc, "_random_projection"):
                save_dict["encoder_random_projection"] = enc._random_projection
            if hasattr(enc, "_inhibition_radius"):
                save_dict["encoder_inhibition_radius"] = np.int64(enc._inhibition_radius)
            if hasattr(enc, "_inhibition_kernel"):
                save_dict["encoder_inhibition_kernel"] = enc._inhibition_kernel

        np.savez(path, **save_dict)


    def load_state(self, path):
        """Load full agent state."""
        data = np.load(path, allow_pickle=True)
        # File contains just the weights array
        if isinstance(data, np.ndarray):
            self.weights = data.astype(np.float32, copy=True)
            return

        # npz with multiple fields
        self.weights = np.asarray(data["weights"], dtype=np.float32)
        if "steps_done" in data.files:
            self.steps_done = int(data["steps_done"])
        if "rng_state" in data.files:
            rng_state_arr = data["rng_state"]
            # rng_state_arr is stored as an object array; convert back to tuple.
            self.rng.set_state(tuple(rng_state_arr.tolist()))

        # Restore encoder parameters if present
        enc = getattr(self, "encoder", None)
        if enc is not None and hasattr(enc, "__dict__"):
            if "encoder_dimension_gains" in data.files:
                enc._dimension_gains = np.asarray(data["encoder_dimension_gains"], dtype=np.float32)
            if "encoder_detector_centers" in data.files:
                enc._detector_centers = np.asarray(data["encoder_detector_centers"], dtype=np.float32)
            if "encoder_random_projection" in data.files:
                enc._random_projection = np.asarray(data["encoder_random_projection"], dtype=np.float32)
            if "encoder_inhibition_radius" in data.files:
                enc._inhibition_radius = int(data["encoder_inhibition_radius"])
            if "encoder_inhibition_kernel" in data.files:
                enc._inhibition_kernel = np.asarray(data["encoder_inhibition_kernel"], dtype=np.float32)
