import os
import sys
from collections import deque
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from bridge_env import BridgeEnv
from neural_sheet_encoder import NeuralSheetEncoder
from bridge_neural_sheet_agent import NeuralSheetQAgent
from settings import InputDimensionCount


class Tee:
    """Stream-like object that forwards every write to multiple underlying streams."""

    def __init__(self, *streams):
        self._streams = streams


    def write(self, data):
        for s in self._streams:
            s.write(data)
            s.flush()


    def flush(self):
        for s in self._streams:
            s.flush()


def build_bridge_dimension_gains():
    """Per-input-dimension gain vector for the 69-D bridge observation.

    Layout (69-D bridge input):
      - 9 local_grid cells × 7 one-hot = 63 dims  
      - 2 position dims (normalized row, col)     
      - 2 time-indicator one-hot dims             
      - 2 has-object one-hot dims                 
    """
    gains = np.ones(InputDimensionCount, dtype=np.float32)

    local_len = 9 * 7
    pos_start = local_len
    pos_end = pos_start + 2
    time_start = pos_end
    time_end = time_start + 2
    carry_start = time_end
    carry_end = carry_start + 2

    down_weight = 0
    gains[pos_start:pos_end] = down_weight  # position 
    gains[time_start:time_end] = 1          # time one-hot 
    gains[carry_start:carry_end] = 1        # has-object one-hot 

    return gains


def discretize_obs(obs):
    """Hashable key for first-visit Monte Carlo bookkeeping (NOT masking).

    Uses the *raw* observation fields:
      - local_grid (3x3 ints)
      - agent_pos (row, col ints)
      - time_indicator (0/1)
      - has_object (0/1)
    """
    grid_flat = tuple(np.asarray(obs["local_grid"], dtype=np.int32).flatten().tolist())
    pos_arr = np.asarray(obs["agent_pos"], dtype=np.int32).reshape(-1)
    pos = (int(pos_arr[0]), int(pos_arr[1]))
    t = int(obs["time_indicator"])
    o = int(obs["has_object"])
    return (grid_flat, pos, t, o)


@dataclass


class TrainConfig:
    # Training
    episodes: int = 100_000
    eval_episodes: int = 200
    seed: int = 111

    # Agent hyperparameters
    learning_rate: float = 0.1
    discount: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay_steps: int = 50_000

    # Monte Carlo update
    first_visit: bool = True
    return_scale: float = 1000.0  # scale rewards for numerical stability

    # Success replay (helps retain rare successful trajectories in sparse reward settings).
    use_success_replay: bool = True
    success_buffer_size: int = 200
    success_replay_updates_per_episode: int = 10

    # Logging / outputs
    log_interval: int = 100
    check_every: int = 1000  # 0 disables early stopping based on greedy evaluation
    output_dir: str = "outputs"


# ----------------------------------------------------------------------
# Off-policy replay of fixed successful trajectories (from .ofp files)
# ----------------------------------------------------------------------

OffPolicyMorning: Optional[List[str]] = None
OffPolicyEvening: Optional[List[str]] = None
OffPolicyUpdateInterval = 30  # episodes between off-policy updates 


def load_traj_file(filename):
    """Load expert trajectories from a plain text file of actions.

    Each non-empty line is interpreted as a sequence of L/R/U/D characters,
    corresponding to left/right/up/down actions. Any other characters are
    discarded.
    """
    path = os.path.join(os.path.dirname(__file__), filename)
    if not os.path.isfile(path):
        return []
    with open(path, "r") as f:
        lines = [ln.strip().upper() for ln in f.readlines()]
    # Keep only lines that contain at least one LRUD character
    clean = []
    for ln in lines:
        filtered = "".join(ch for ch in ln if ch in "LRUD")
        if filtered:
            clean.append(filtered)
    return clean


def parse_action_char(ch):
    """Map an action character to the discrete action index used by BridgeEnv."""
    mapping = {"L": 0, "R": 1, "U": 2, "D": 3}
    return mapping.get(ch, None)


def offpolicy_from_actions(env_template, agent, cfg, actions, t_flag):
    """Replay a fixed action sequence in a fresh environment and update the agent.

    The fixed action sequence is interpreted as an expert trajectory, and a
    first-visit Monte-Carlo regression update is applied along that trajectory.
    """
    # Fresh environment with same basic configuration
    test_env = BridgeEnv(grid_size=env_template.grid_size, render_mode=None, phase=env_template.phase)
    obs, _ = test_env.reset()

    # Fix time-of-day and object position to match the desired scenario
    test_env.time_of_day = int(t_flag)
    test_env.object_pos = (test_env.object_loc_morning if t_flag == 0 else test_env.object_loc_evening)
    test_env.grid = test_env._init_grid()
    obs = test_env._get_obs()

    traj_keys: List[Tuple] = []
    traj_phi: List[np.ndarray] = []
    traj_act: List[int] = []
    traj_rew_scaled: List[float] = []

    done = False
    truncated = False

    for ch in actions:
        if done or truncated:
            break
        a = parse_action_char(ch)
        if a is None:
            continue

        key = discretize_obs(obs)
        phi = agent.features_from_obs(obs)

        next_obs, reward, done, truncated, _ = test_env.step(int(a))

        traj_keys.append(key)
        traj_phi.append(phi)
        traj_act.append(int(a))
        traj_rew_scaled.append(float(reward) / float(cfg.return_scale))

        obs = next_obs

    if not traj_act:
        return

    # First-visit Monte Carlo update over the replayed trajectory
    G = 0.0
    visited_sa = set()
    for t in reversed(range(len(traj_act))):
        G = traj_rew_scaled[t] + float(cfg.discount) * G
        if cfg.first_visit:
            sa = (traj_keys[t], int(traj_act[t]))
            if sa in visited_sa:
                continue
            visited_sa.add(sa)

        agent.update_mc(traj_phi[t], int(traj_act[t]), float(G))


def offpolicy_sample_and_update(env, agent, cfg, rng):
    """Sample one morning and one evening trajectory and update the agent."""
    global OffPolicyMorning, OffPolicyEvening

    if OffPolicyMorning is None:
        OffPolicyMorning = load_traj_file("morning.ofp")
    if OffPolicyEvening is None:
        OffPolicyEvening = load_traj_file("evening.ofp")

    if OffPolicyMorning:
        idx = int(rng.randint(len(OffPolicyMorning)))
        offpolicy_from_actions(env, agent, cfg, OffPolicyMorning[idx], t_flag=0)

    if OffPolicyEvening:
        idx = int(rng.randint(len(OffPolicyEvening)))
        offpolicy_from_actions(env, agent, cfg, OffPolicyEvening[idx], t_flag=1)


def run_training(cfg):
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Mirror all console output to a log file in the output directory.
    log_path = os.path.join(cfg.output_dir, "console_output.txt")

    # Install the tee only once per process to avoid duplicate writes if run_training
    # is invoked multiple times.
    if not hasattr(run_training, "_tee_installed"):
        log_file = open(log_path, "a", encoding="utf-8")
        sys.stdout = Tee(sys.stdout, log_file)
        sys.stderr = Tee(sys.stderr, log_file)
        run_training._tee_installed = True

    # State file path (new name) and legacy fallbacks for backward compatibility
    state_path = os.path.join(cfg.output_dir, "program_state.npz")

    env = BridgeEnv(render_mode=None, phase="train")
    env.reset(seed=cfg.seed)

    dim_gains = build_bridge_dimension_gains()
    encoder = NeuralSheetEncoder(random_seed=cfg.seed, dimension_gains=dim_gains)
    agent = NeuralSheetQAgent(
        encoder=encoder,
        num_actions=env.action_space.n,
        learning_rate=cfg.learning_rate,
        discount=cfg.discount,
        epsilon_start=cfg.epsilon_start,
        epsilon_end=cfg.epsilon_end,
        epsilon_decay_steps=cfg.epsilon_decay_steps,
        seed=cfg.seed,
        normalize_features=True,
        td_clip=1.0,
        weight_clip=200.0,
    )

    loaded_from = None
    for candidate in (state_path):
        if os.path.isfile(state_path):
            agent.load_state(state_path)
            loaded_from = state_path
            break

    if loaded_from is not None:
        print(f"Loaded weights from {loaded_from}")
    else:
        print("No existing weights found. Training from scratch.")

    # Buffer of successful episode update lists: each element is List[(phi16, action, G)]
    success_buffer: deque = deque(maxlen=int(cfg.success_buffer_size))

    episode_returns: List[float] = []
    episode_success: List[float] = []

    for ep in range(1, int(cfg.episodes) + 1):
        obs, info = env.reset()
        done = False
        truncated = False

        traj_keys: List[Tuple] = []
        traj_phi: List[np.ndarray] = []
        traj_action: List[int] = []
        traj_reward_scaled: List[float] = []

        total_reward = 0.0
        last_info: Dict[str, Any] = {"is_success": False}

        while not (done or truncated):
            key = discretize_obs(obs) if cfg.first_visit else None
            action, phi = agent.select_action(obs, greedy=False)

            next_obs, reward, done, truncated, info = env.step(action)

            traj_keys.append(key if key is not None else ())
            traj_phi.append(phi)
            traj_action.append(int(action))
            traj_reward_scaled.append(float(reward) / float(cfg.return_scale))

            total_reward += float(reward)
            obs = next_obs
            last_info = info

        success = bool(last_info.get("is_success", False))
        episode_returns.append(total_reward)
        episode_success.append(1.0 if success else 0.0)

        # Monte Carlo returns and updates
        G = 0.0
        visited_sa = set()
        updates: List[Tuple[np.ndarray, int, float]] = []

        for t in reversed(range(len(traj_action))):
            G = traj_reward_scaled[t] + float(cfg.discount) * G

            if cfg.first_visit:
                sa = (traj_keys[t], int(traj_action[t]))
                if sa in visited_sa:
                    continue
                visited_sa.add(sa)

            a_t = int(traj_action[t])
            phi_t = traj_phi[t]
            agent.update_mc(phi_t, a_t, G)

            if cfg.use_success_replay:
                updates.append((phi_t.astype(np.float16, copy=False), a_t, float(G)))

        # Store successful trajectories for replay
        if cfg.use_success_replay and success and len(updates) > 0:
            success_buffer.append(updates)

        # Replay: sample a few stored successful updates each episode (no masking)
        if cfg.use_success_replay and len(success_buffer) > 0 and int(cfg.success_replay_updates_per_episode) > 0:
            for _ in range(int(cfg.success_replay_updates_per_episode)):
                ep_updates = success_buffer[int(agent.rng.randint(len(success_buffer)))]
                phi16, a, Gt = ep_updates[int(agent.rng.randint(len(ep_updates)))]
                agent.update_mc(phi16.astype(np.float32, copy=False), int(a), float(Gt))

        # Off-policy replay from fixed morning/evening expert trajectories
        if ep % int(OffPolicyUpdateInterval) == 0:
            offpolicy_sample_and_update(env, agent, cfg, agent.rng)

        if ep % int(cfg.log_interval) == 0:
            mean_ret = float(np.mean(episode_returns[-int(cfg.log_interval):]))
            mean_succ = float(np.mean(episode_success[-int(cfg.log_interval):]))
            print(f"Episode {ep:5d} | mean return {mean_ret:8.1f} | success rate {mean_succ * 100:5.1f}%")

        if int(getattr(cfg, "check_every", 0)) > 0 and ep % int(cfg.check_every) == 0:
            all_success = evaluate_and_visualize(agent, encoder, cfg)
            if all_success:
                print(f"All 4 evaluation scenarios successful at episode {ep}. Stopping early.")
                break

    # Save weights/state to the new consolidated state file
    state_path = os.path.join(cfg.output_dir, "program_state.npz")
    agent.save_state(state_path)
    print(f"Saved weights to {state_path}")

    # Evaluate and visualize
    evaluate_and_visualize(agent, encoder, cfg)


def evaluate_and_visualize(agent, encoder, cfg):
    """Greedy evaluation on four fixed scenarios: morning/evening × train/test.

    Returns True if all four scenarios are successful under the greedy policy,
    and False otherwise.
    """
    all_success = True


    def run_case(phase, time_flag):
        nonlocal all_success

        env = BridgeEnv(render_mode=None, phase=phase)
        obs, info = env.reset()

        # Fix time of day and corresponding object location
        env.time_of_day = int(time_flag)
        env.object_pos = env.object_loc_morning if time_flag == 0 else env.object_loc_evening
        env.grid = env._init_grid()
        obs = env._get_obs()

        coords: List[Tuple[int, int]] = []
        steps = 0
        done = False
        truncated = False
        max_steps = env.max_steps

        # Record initial position
        r, c = obs["agent_pos"]
        coords.append((int(c), int(r)))

        last_info: Dict[str, Any] = {"is_success": False}

        while not (done or truncated) and steps < max_steps:
            action, phi = agent.select_action(obs, greedy=True)
            obs, reward, done, truncated, info = env.step(action)
            steps += 1
            r, c = obs["agent_pos"]
            coords.append((int(c), int(r)))
            last_info = info

        success = bool(last_info.get("is_success", False))
        if not success:
            all_success = False
        label_time = "morning" if time_flag == 0 else "evening"
        print(f"phase={phase}, time={label_time}")
        print(f"steps={steps}")
        print(f"trajectory={coords}")
        print(f"success={success}")
        print()

    for phase in ["train", "test"]:
        for t_flag in [0, 1]:
            run_case(phase, t_flag)

    return all_success


if __name__ == "__main__":
    run_training(TrainConfig())
