import pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_summaries(runs_dir="runs"):
    data = {}
    for strategy_dir in Path(runs_dir).iterdir():
        if not strategy_dir.is_dir():
            continue
        for f in strategy_dir.glob("summary_*.pkl"):
            try:
                with open(f, "rb") as fh:
                    summ = pickle.load(fh)
                d = summ[0] if isinstance(summ, list) and summ else summ
                key = f"{strategy_dir.name}/{f.name.replace('.pkl','')}"
                data[key] = d
            except Exception as e:
                print(f"[WARN] Could not read {f}: {e}")
    return data

def plot_learning_curves(summary_data, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Learning Curves Comparison (per Epoch)", fontsize=16, fontweight="bold")
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(summary_data))))
    plotted = False
    for (name, d), color in zip(sorted(summary_data.items()), colors):
        train_losses = d.get("train_loss", [])
        val_losses = d.get("val_loss", [])
        val_accs = d.get("val_acc", [])
        val_bal_accs = d.get("val_bal_acc", [])
        if not train_losses:
            continue
        epochs = range(1, len(train_losses) + 1)
        label = name
        axes[0, 0].plot(epochs, train_losses, label=label, color=color, linewidth=2)
        if val_losses:
            axes[0, 1].plot(epochs, val_losses, label=label, color=color, linewidth=2)
        if val_accs:
            axes[1, 0].plot(epochs, val_accs, label=label, color=color, linewidth=2)
        if val_bal_accs:
            axes[1, 1].plot(epochs, val_bal_accs, label=label, color=color, linewidth=2)
        plotted = True
    axes[0, 0].set_title("Training Loss (per Epoch)");    axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("Loss")
    axes[0, 1].set_title("Validation Loss (per Epoch)");  axes[0, 1].set_xlabel("Epoch"); axes[0, 1].set_ylabel("Loss")
    axes[1, 0].set_title("Validation Accuracy (per Epoch)"); axes[1, 0].set_xlabel("Epoch"); axes[1, 0].set_ylabel("Accuracy")
    axes[1, 1].set_title("Validation Balanced Accuracy (per Epoch)"); axes[1, 1].set_xlabel("Epoch"); axes[1, 1].set_ylabel("Balanced Accuracy")
    for ax in axes.flat:
        ax.grid(True, alpha=0.3); ax.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if plotted:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Learning curves saved to: {save_path}")
    else:
        print("No data to plot.")
    plt.show()


def plot_all_metrics(summary_data, save_path):
    all_metrics = set()
    for d in summary_data.values():
        for k, v in d.items():
            if isinstance(v, list) and len(v) > 1:
                all_metrics.add(k)
    all_metrics = sorted(all_metrics)

    n_metrics = len(all_metrics)
    ncols = 2
    nrows = (n_metrics + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows))
    axes = axes.flatten() if n_metrics > 1 else [axes]
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(summary_data))))
    plotted = False

    for i, metric in enumerate(all_metrics):
        ax = axes[i]
        for (name, d), color in zip(sorted(summary_data.items()), colors):
            values = d.get(metric, [])
            if values:
                epochs = range(1, len(values) + 1)
                ax.plot(epochs, values, label=name, color=color, linewidth=2)
                plotted = True
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        ax.legend()
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    if plotted:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"All metrics curves saved to: {save_path}")
    else:
        print("No data to plot.")
    plt.show()

def main():
    # Save plots in analysis/plots/
    PLOTS_DIR = Path(__file__).parent / "plots"
    PLOTS_DIR.mkdir(exist_ok=True)
    runs_dir = str((Path(__file__).parent.parent / "runs").resolve())
    summary_data = load_summaries(runs_dir)
    if not summary_data:
        print("No summary files found."); return
    plot_path = PLOTS_DIR / "all_metrics.png"
    plot_all_metrics(summary_data, save_path=str(plot_path))

if __name__ == "__main__":
    main()


