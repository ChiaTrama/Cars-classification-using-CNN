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

def extract_difficulty(summary_key):
    for diff in ["easy", "medium", "hard"]:
        if diff in summary_key:
            return diff
    return "unknown"

def plot_verification_metrics(summary_data, save_dir):
    verification_runs = {k: v for k, v in summary_data.items() if v.get("task") == "verification"}
    if not verification_runs:
        print("No verification runs found."); return

    all_metrics = set()
    for d in verification_runs.values():
        for k, v in d.items():
            if isinstance(v, list) and len(v) > 1:
                all_metrics.add(k)
    all_metrics = sorted(all_metrics)


    if "train_loss" in all_metrics and "val_loss" in all_metrics:
        plt.figure(figsize=(8, 5))
        colors = plt.cm.tab10(np.linspace(0, 1, len(verification_runs)))
        for (key, d), color in zip(sorted(verification_runs.items()), colors):
            strat_name = d.get("name", key.split("/")[0])
            difficulty = extract_difficulty(key)
            label_train = f"{strat_name} - {difficulty} - train"
            label_val = f"{strat_name} - {difficulty} - val"
            epochs = range(1, len(d.get("train_loss", [])) + 1)
            plt.plot(epochs, d.get("train_loss", []), label=label_train, color=color, linestyle="-")
            plt.plot(epochs, d.get("val_loss", []), color=color, linestyle="--")
        plt.title("Train & Validation Loss (Verification)")
        plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.grid(True, alpha=0.3); plt.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/verification_train_val_loss.png", dpi=300)
        plt.close()

    for metric in all_metrics:
        if metric in ["train_loss", "val_loss"]:
            continue
        plt.figure(figsize=(8, 5))
        colors = plt.cm.tab10(np.linspace(0, 1, len(verification_runs)))
        for (key, d), color in zip(sorted(verification_runs.items()), colors):
            strat_name = d.get("name", key.split("/")[0])
            difficulty = extract_difficulty(key)
            label = f"{strat_name} - {difficulty}"
            values = d.get(metric, [])
            if values:
                epochs = range(1, len(values) + 1)
                plt.plot(epochs, values, label=label, color=color, linewidth=2)
        plt.title(f"{metric.replace('_', ' ').title()} (Verification)")
        plt.xlabel("Epoch"); plt.ylabel(metric)
        plt.grid(True, alpha=0.3); plt.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/verification_{metric}.png", dpi=300)
        plt.close()
    print(f"Verification metrics plots saved in {save_dir}")

def main():
    PLOTS_DIR = Path(__file__).parent / "plots"
    PLOTS_DIR.mkdir(exist_ok=True)
    runs_dir = str((Path(__file__).parent.parent / "runs").resolve())
    summary_data = load_summaries(runs_dir)
    if not summary_data:
        print("No summary files found."); return
    plot_verification_metrics(summary_data, save_dir=str(PLOTS_DIR))

if __name__ == "__main__":
    main()