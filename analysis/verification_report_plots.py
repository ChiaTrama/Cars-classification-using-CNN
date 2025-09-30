from pathlib import Path
import pickle
import matplotlib.pyplot as plt

# Directory base
base_dir = Path(__file__).parent.parent  # project_cars/
runs_dir = base_dir / "runs"
plots_dir = Path(__file__).parent / "plots"
plots_dir.mkdir(exist_ok=True)

paths = [
    runs_dir / "SiameseEfficientNet_B0_192x192/summary_make_contrastive_bs64_ep75_easy.pkl",
    runs_dir / "SiameseResNet18_192x192/summary_make_contrastive_bs128_ep75_easy.pkl",
    runs_dir / "SiameseResNet50_30M_192x192_pretrained/summary_make_contrastive_bs64_ep75_easy.pkl",
    runs_dir / "SiameseResNet18_192x192_scratch/summary_make_contrastive_bs128_ep75_easy.pkl",
]
labels = ["EfficientNet", "ResNet18", "ResNet50", "ResNet18 Scratch"]

colors = plt.get_cmap("tab10").colors  

plt.figure(figsize=(8,5))
for i, (path, label) in enumerate(zip(paths, labels)):
    with open(path, "rb") as f:
        summary = pickle.load(f)
    if isinstance(summary, list):
        summary = summary[0]
    train_loss = summary["train_loss"][:45]
    val_loss = summary["val_loss"][:45]
    plt.plot(train_loss, label=f"{label} - train", color=colors[i])
    plt.plot(val_loss, label=f"{label} - val", color=colors[i], linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.ylim(0, 1)
plt.title("Training & Validation Loss (Siamese Models)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("analysis/plots/siamese_models_train_val_loss.png", dpi=300)
plt.close()


paths = [
    runs_dir / "SiameseResNet18_192x192/summary_make_contrastive_bs128_ep75_easy.pkl",
    runs_dir / "SiameseResNet18_192x192/summary_make_contrastive_bs128_ep45_medium.pkl",
    runs_dir / "SiameseResNet18_192x192/summary_make_contrastive_bs64_ep45_hard.pkl",
    runs_dir / "SiameseResNet18_192x192/summary_model_contrastive_bs128_ep45_easy.pkl",
    runs_dir / "SiameseResNet18_192x192/summary_model_contrastive_bs128_ep45_medium.pkl",
    runs_dir / "SiameseResNet18_192x192/summary_model_contrastive_bs128_ep45_hard.pkl"
]
labels = [
    "Make - Easy", "Make - Medium", "Make - Hard",
    "Model - Easy", "Model - Medium", "Model - Hard"
]

plt.figure(figsize=(8,5))
for path, label in zip(paths, labels):
    with open(path, "rb") as f:
        summary = pickle.load(f)
    if isinstance(summary, list):
        summary = summary[0]
    val_acc = summary["val_acc"][:45]
    if "Model" in label:
        plt.plot(val_acc, label=label, linestyle="--")
    else:
        plt.plot(val_acc, label=label)
plt.xlabel("Epoch")
plt.ylabel("Val Accuracy")
plt.title("Validation Accuracy (SiameseResNet18)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(plots_dir / "siamese_resnet18_val_acc.png", dpi=300)
plt.close()

plt.figure(figsize=(8,5))
for path, label in zip(paths, labels):
    with open(path, "rb") as f:
        summary = pickle.load(f)
    if isinstance(summary, list):
        summary = summary[0]
    val_roc_auc = summary["val_roc_auc"][:45]
    plt.plot(val_roc_auc, label=label)
plt.xlabel("Epoch")
plt.ylabel("Val ROC AUC")
plt.title("Validation ROC AUC (SiameseResNet18)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(plots_dir / "siamese_resnet18_val_roc_auc.png", dpi=300)
plt.close()


from sklearn.metrics import roc_curve, auc

summary_files = [
    "runs/SiameseResNet18_192x192/summary_make_contrastive_bs128_ep2_easy.pkl",
    "runs/SiameseResNet18_192x192/summary_make_contrastive_bs128_ep20_easy.pkl",
    "runs/SiameseResNet18_192x192/summary_make_contrastive_bs128_ep45_easy.pkl"
]
labels = ["easy", "easy", "easy"]

plt.figure(figsize=(7,7))
epochs_labels = [
    "Epoch 2 (AUC={:.2f})",
    "Epoch 20 (AUC={:.2f})",
    "Epoch 45 (AUC={:.2f})"
]
for path, label_fmt in zip(summary_files, epochs_labels):
    with open(path, "rb") as f:
        summary = pickle.load(f)
    if isinstance(summary, list):
        summary = summary[0]
    dists = summary["all_dists"]
    y_true = summary["all_labels"]
    if hasattr(dists, "cpu"):
        dists = dists.cpu().numpy()
    if hasattr(y_true, "cpu"):
        y_true = y_true.cpu().numpy()
    scores = -dists
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=label_fmt.format(roc_auc))

plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random Classifier')
plt.plot([0, 0, 1], [0, 1, 1], color='black', lw=1.5, linestyle=':', label='Perfect Classifier')
plt.xlim([-0.02, 1.0])
plt.ylim([0.0, 1.02])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC curve – Verification (easy)')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("analysis/plots/roc_curve_verification_easy.png", dpi=300)


summary_files = [
    "runs/SiameseResNet18_192x192/summary_make_contrastive_bs128_ep20_easy.pkl",
    "runs/SiameseResNet18_192x192/summary_make_contrastive_bs128_ep20_medium.pkl",
    "runs/SiameseResNet18_192x192/summary_make_contrastive_bs64_ep20_hard.pkl"
]
labels = ["easy", "medium", "hard"]

plt.figure(figsize=(7,7))
epochs_labels = [
    "Easy (AUC={:.2f})",
    "Medium (AUC={:.2f})",
    "Hard (AUC={:.2f})"
]
for path, label_fmt in zip(summary_files, epochs_labels):
    with open(path, "rb") as f:
        summary = pickle.load(f)
    if isinstance(summary, list):
        summary = summary[0]
    dists = summary["all_dists"]
    y_true = summary["all_labels"]
    if hasattr(dists, "cpu"):
        dists = dists.cpu().numpy()
    if hasattr(y_true, "cpu"):
        y_true = y_true.cpu().numpy()
    scores = -dists
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=label_fmt.format(roc_auc))

plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random Classifier')
plt.plot([0, 0, 1], [0, 1, 1], color='black', lw=1.5, linestyle=':', label='Perfect Classifier')
plt.xlim([-0.02, 1.0])
plt.ylim([0.0, 1.02])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC curve – Verification (20 epochs)')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("analysis/plots/roc_curve_verification_20ep.png", dpi=300)