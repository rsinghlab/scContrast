import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Manually enter the results as a nested dictionary
results = {
    "Epochs=1": {
        "raised_power=1":      [0.8453, 0.8170, 0.5830, 0.4301, 0.6929, 0.9609, 0.9220, 0.8269, 0.7195, 0.7695],
        "raised_power=1.5":    [0.8340, 0.7824, 0.5758, 0.3640, 0.6590, 0.9823, 0.9195, 0.8206, 0.6952, 0.7527],
        "raised_power=2":      [0.7874, 0.7467, 0.5244, 0.3560, 0.5007, 0.8643, 0.9066, 0.6825, 0.6642, 0.6732],
        "softmax":             [0.8347, 0.7738, 0.5627, 0.4116, 0.7024, 0.9652, 0.9164, 0.8338, 0.6998, 0.7609],
    },
    "Epochs=50": {
        "raised_power=1":      [0.8690, 0.8533, 0.5880, 0.4535, 0.7419, 0.9720, 0.9412, 0.8570, 0.7410, 0.7948],
        "raised_power=1.5":    [0.8565, 0.8233, 0.5939, 0.4653, 0.7298, 0.9781, 0.9365, 0.8539, 0.7351, 0.7901],
        "raised_power=2":      [0.8521, 0.8366, 0.5893, 0.4386, 0.7124, 0.9735, 0.9299, 0.8430, 0.7293, 0.7820],
        "softmax":             [0.8260, 0.8003, 0.5657, 0.4093, 0.6909, 0.9749, 0.9118, 0.8329, 0.7026, 0.7622],
    }
}

# Abbreviated and multi-line metric names for better alignment
main_metric_names = [
    "NMI", "ARI", "Label\nASW", "Isolated\nF1", "Batch\nASW", "PCR\nbatch", "Graph\nconn"
]
overall_metric_names = [
    "Overall\nbatch", "Overall\nbio", "Overall\nscore"
]

methods = list(results["Epochs=1"].keys())

# Prepare data for each epoch and for main/overall metrics
def split_metrics(epoch_data):
    main_data = []
    overall_data = []
    for method in methods:
        vals = epoch_data[method]
        main_data.append(vals[:-3])
        overall_data.append(vals[-3:])
    return np.array(main_data), np.array(overall_data)

main_data_1, overall_data_1 = split_metrics(results["Epochs=1"])
main_data_50, overall_data_50 = split_metrics(results["Epochs=50"])

fig, axes = plt.subplots(2, 2, figsize=(18, 12), gridspec_kw={'height_ratios': [3, 1]})

# Main metrics heatmaps (top row)
sns.heatmap(main_data_1, annot=True, fmt=".3f", cmap="YlGnBu",
            xticklabels=main_metric_names, yticklabels=methods, ax=axes[0, 0])
axes[0, 0].set_title("Epochs=1: Main Metrics")
axes[0, 0].set_ylabel("Method")
axes[0, 0].set_xlabel("Metric")
axes[0, 0].tick_params(axis='x', rotation=0)

sns.heatmap(main_data_50, annot=True, fmt=".3f", cmap="YlGnBu",
            xticklabels=main_metric_names, yticklabels=methods, ax=axes[0, 1])
axes[0, 1].set_title("Epochs=50: Main Metrics")
axes[0, 1].set_ylabel("")
axes[0, 1].set_xlabel("Metric")
axes[0, 1].tick_params(axis='x', rotation=0)

# Overall metrics heatmaps (bottom row)
sns.heatmap(overall_data_1, annot=True, fmt=".3f", cmap="YlOrRd",
            xticklabels=overall_metric_names, yticklabels=methods, ax=axes[1, 0], cbar=False)
axes[1, 0].set_title("Epochs=1: Overall Metrics")
axes[1, 0].set_ylabel("Method")
axes[1, 0].set_xlabel("Overall Metric")
axes[1, 0].tick_params(axis='x', rotation=0)

sns.heatmap(overall_data_50, annot=True, fmt=".3f", cmap="YlOrRd",
            xticklabels=overall_metric_names, yticklabels=methods, ax=axes[1, 1], cbar=False)
axes[1, 1].set_title("Epochs=50: Overall Metrics")
axes[1, 1].set_ylabel("")
axes[1, 1].set_xlabel("Overall Metric")
axes[1, 1].tick_params(axis='x', rotation=0)

plt.tight_layout(pad=2.0)
plt.subplots_adjust(wspace=0.25, hspace=0.35)
plt.savefig("scib_metrics_combined_heatmaps.png", dpi=300)
print("Combined heatmaps saved to scib_metrics_combined_heatmaps.png.")