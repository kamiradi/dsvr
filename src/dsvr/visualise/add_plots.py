import matplotlib.pyplot as plt

def plot_add_vs_time(times, add_mean, add_min, add_best):
    plt.figure(figsize=(7, 4))

    plt.plot(times, add_mean, label="Mean ADD", linewidth=2)
    plt.plot(times, add_best, label="Best-particle ADD", linewidth=2)
    plt.plot(times, add_min, label="Min ADD", linestyle="--")

    plt.xlabel("Time (s)")
    plt.ylabel("ADD")
    plt.title("ADD vs Time")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
