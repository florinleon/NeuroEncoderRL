import numpy as np
import matplotlib.pyplot as plt


def load_metrics(path):
    # Detect delimiter from the first line
    with open(path, "r") as f:
        first_line = f.readline()
    if "," in first_line:
        data = np.loadtxt(path, delimiter=",")
    else:
        data = np.loadtxt(path)
    episodes = data[:, 0]
    mean_return = data[:, 1]
    success = data[:, 2]
    return episodes, mean_return, success


def moving_average_naive(values, step_in_episodes, window_in_episodes=1000.0):
    if step_in_episodes <= 0:
        window_points = 1
    else:
        window_points = int(window_in_episodes / step_in_episodes)
        if window_points < 1:
            window_points = 1

    n = len(values)
    smoothed = np.zeros(n, dtype=float)
    for i in range(n):
        start = max(0, i - window_points + 1)
        s = 0.0
        count = 0
        for j in range(start, i + 1):
            s += values[j]
            count += 1
        smoothed[i] = s / count
    return smoothed


def main():
    paths = [
        "console_output_episode_metrics1.txt",
        "console_output_episode_metrics2.txt",
        "console_output_episode_metrics3.txt",
    ]

    runs = [load_metrics(p) for p in paths]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    labels = ["Run 1", "Run 2", "Run 3"]

    # Figure 1: mean return
    fig1, ax1 = plt.subplots(figsize=(10, 5))

    for (episodes, mean_return, success), color, label in zip(runs, colors, labels):
        diffs = np.diff(episodes)
        step = float(np.median(diffs)) if len(diffs) > 0 else 1.0
        smooth_mean = moving_average_naive(mean_return, step_in_episodes=step, window_in_episodes=1000.0)
        ax1.plot(episodes, mean_return, color=color, alpha=0.25, linewidth=1)
        ax1.plot(episodes, smooth_mean, color=color, alpha=1.0, linewidth=2, label=label)

    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Mean return")
    ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax1.legend()

    # Figure 2: success rate
    # fig2, ax2 = plt.subplots(figsize=(10, 5))

    # for (episodes, mean_return, success), color, label in zip(runs, colors, labels):
    #     diffs = np.diff(episodes)
    #     step = float(np.median(diffs)) if len(diffs) > 0 else 1.0
    #     smooth_success = moving_average_naive(success, step_in_episodes=step, window_in_episodes=1000.0)
    #     ax2.plot(episodes, success, color=color, alpha=0.25, linewidth=1)
    #     ax2.plot(episodes, smooth_success, color=color, alpha=1.0, linewidth=2, label=label)

    # ax2.set_xlabel("Episode")
    # ax2.set_ylabel("Success rate (%)")
    # ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    # ax2.legend()

    #plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
