import json
import numpy as np
import matplotlib.pyplot as plt


def plot_calibration(results_path: str, output_path: str, n_bins: int = 10):
    with open(results_path) as f:
        data = json.load(f)

    y_scores = np.array(data.get("y_scores", []))
    y_true = np.array(data.get("y_true", []))

    if len(y_scores) == 0:
        print("No scores to calibrate.")
        return

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers, bin_accuracies = [], []

    for i in range(n_bins):
        mask = (y_scores >= bin_edges[i]) & (y_scores < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_accuracies.append(y_true[mask].mean())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax1.bar(bin_centers, bin_accuracies, width=1 / n_bins, alpha=0.7, edgecolor="black")
    ax1.set_xlabel("Predicted hallucination probability")
    ax1.set_ylabel("Actual hallucination rate")
    ax1.set_title("Calibration (Reliability Diagram)")
    ax1.legend()

    ax2.hist(y_scores, bins=n_bins, edgecolor="black", alpha=0.7)
    ax2.set_xlabel("Predicted hallucination probability")
    ax2.set_ylabel("Count")
    ax2.set_title("Prediction Distribution")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Calibration plot saved to {output_path}")
