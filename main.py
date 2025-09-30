import os
import time
import argparse
import pickle
import torch
from src.dataset import CompCarsDataset, get_albu_transform, CachedDataset
from src.dataset_verification import CompCarsVerificationDataset, SiameseDataset, CompCarsBaseDataset
from src.train import train, train_verification
from src.strategies import get_strategies, serialize_strategy
import questionary

def get_verification_datasets(image_path, base_path, target_size, target="model"):
    from src.dataset_verification import CompCarsVerificationDataset, CompCarsBaseDataset, SiameseDataset
    from src.dataset import get_albu_transform

    verification_dir = os.path.join(base_path, "train_test_split", "verification")
    train_list_file = os.path.join(verification_dir, "verification_train.txt")

    val_level = questionary.select(
        "Select validation difficulty level:",
        choices=["easy", "medium", "hard"]
    ).ask()
    val_pairs_file = os.path.join(
        verification_dir, f"verification_pairs_{val_level}.txt"
    )

    # Dataset base for training
    base_train_dataset = CompCarsBaseDataset(
        image_list_file=train_list_file,
        label_dir=os.path.join(base_path, "label"),
        image_dir=image_path,
        target=target,
        target_size=target_size,
        augmentations=get_albu_transform(target_size=target_size, train=True)
    )
    # SiameseDataset that generates pairs on-the-fly
    train_dataset = SiameseDataset(base_train_dataset)

    # Validation dataset with pre-made pairs
    val_dataset = CompCarsVerificationDataset(
        pairs_file=val_pairs_file,
        image_dir=image_path,
        target_size=target_size,
        augmentations=get_albu_transform(target_size=target_size, train=False),
        train=False
    )
    return train_dataset, val_dataset, val_level

# Basic configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_PATH = "CompCars/data/data"
IMAGE_PATH = os.path.join(BASE_PATH, "image")
LABEL_PATH = os.path.join(BASE_PATH, "label")
SPLIT_PATH = os.path.join(BASE_PATH, "train_test_split", "classification")
train_split_file = os.path.join(SPLIT_PATH, "train.txt")
test_split_file = os.path.join(SPLIT_PATH, "test.txt")
RUNS_DIR = "runs"

# Training parameters
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 2e-5
LABEL_SMOOTHING = 0.1  # For cross-entropy loss
NUM_WORKERS = 4  # Adjust based on your system capabilities
CACHE_TEST_DATASET = True
EARLY_STOPPING_PATIENCE = 150

# Argument parser for strategy, target, and loss_type selection
parser = argparse.ArgumentParser()
parser.add_argument("--strategy", type=str, required=False, help="Name of the strategy to use")
parser.add_argument("--target", type=str, choices=["make", "model"], required=False, help="Classification target: 'make' or 'model'")
parser.add_argument("--loss_type", type=str, choices=["crossentropy", "contrastive"], required=False, help="Loss type")
parser.add_argument("--epochs", type=int, required=False, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, required=False, help="Batch size for training")
args = parser.parse_args()

# Select task to do:
task = questionary.select(
    "Select task type:",
    choices=["classification", "verification"]
).ask()

# 1. Select target (make/model) first
if not args.target:
    target = questionary.select(
        "Select classification target:",
        choices=["make", "model"]
    ).ask()
    if not target:
        print("No target selected. Exiting.")
        exit(1)
else:
    target = args.target

# 
if task == "verification":
    NUM_CLASSES = None  # Not used in verification
    if CACHE_TEST_DATASET is True:
        print("Warning: Caching test dataset in memory may use a lot of RAM.")
        CACHE_TEST_DATASET = questionary.select(
            "Select if you want to cache the test dataset in memory:",
            choices=["yes", "no"]
        ).ask()
        if CACHE_TEST_DATASET == "no":
            CACHE_TEST_DATASET = False
    loss_type = "contrastive"


elif task == "classification":
    from src.dataset import CompCarsDataset

    # 2. Load dataset once to get NUM_CLASSES for the selected target
    print("Loading dataset to determine NUM_CLASSES...")
    tmp_dataset = CompCarsDataset(
        split_file=train_split_file,
        image_dir=IMAGE_PATH,
        label_dir=LABEL_PATH,
        target=target,
        use_bbox=False,
        target_size=(224, 224),
        augmentations=get_albu_transform(target_size=(224, 224), train=False),
        train=False
    )
    NUM_CLASSES = tmp_dataset.num_classes
    print(f"NUM_CLASSES for classification: {NUM_CLASSES}")
else:
    print("Invalid task type. Exiting.")
    exit(1)

# 3. Now load strategies with correct NUM_CLASSES
strategies = get_strategies(NUM_CLASSES)

# 4. Select strategy
if not args.strategy or args.strategy not in strategies:
    print("Available strategies:")
    strategy_key = questionary.select(
        "Select a strategy:",
        choices=list(strategies.keys())
    ).ask()
    if not strategy_key:
        print("No strategy selected. Exiting.")
        exit(1)
else:
    strategy_key = args.strategy

# 5. Select loss_type
if not args.loss_type and task == "classification":
    loss_type = questionary.select(
        "Select loss type:",
        choices=["crossentropy", "focal"]
    ).ask()
    if not loss_type:
        print("No loss type selected. Exiting.")
        exit(1)
elif task == "verification":
    loss_type = "contrastive"
else:
    loss_type = args.loss_type

# Select number of epochs
if args.epochs is not None:
    NUM_EPOCHS = args.epochs
else:
    NUM_EPOCHS = int(questionary.text("Number of training epochs?", default="45").ask())

# Select batch size
if not args.batch_size:
    BATCH_SIZE = int(questionary.text("Batch size?", default="64").ask())
else:
    BATCH_SIZE = args.batch_size

strat = strategies[strategy_key]
print("\nSelected configuration:")
print(f"==> Task: {task}")
print(f"==> Using strategy: {strat['name']} ({strategy_key})")
print(f"==> Classification target: {target}")
print(f"==> Loss type: {loss_type}")
print(f"==> Batch size: {BATCH_SIZE}")  
print(f"==> Number of epochs: {NUM_EPOCHS}\n")

print(f"Using {NUM_WORKERS} workers for data loading.")
if CACHE_TEST_DATASET:
    print("Test dataset will be cached in memory (high RAM usage).")

# Prepare paths for this strategy
logger_name = strategy_key
strategy_dir = os.path.join(RUNS_DIR, logger_name)
checkpoint_dir = os.path.join(strategy_dir, "checkpoints")
tb_log_dir = os.path.join(strategy_dir, "tb_logs")
profiler_dir = os.path.join(strategy_dir, "profiler_logs")

os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(tb_log_dir, exist_ok=True)
os.makedirs(profiler_dir, exist_ok=True)

start_time = time.time()
if task == "classification":
    train_dataset = CompCarsDataset(
        split_file=train_split_file,
        image_dir=IMAGE_PATH,
        label_dir=LABEL_PATH,
        target=target,
        use_bbox=strat["use_bbox"],
        target_size=strat["target_size"],
        augmentations=strat["augmentations"],
        train=True,
        use_dataset_mean=strat.get("use_dataset_mean", False)
    )
    test_dataset = CompCarsDataset(
        split_file=test_split_file,
        image_dir=IMAGE_PATH,
        label_dir=LABEL_PATH,
        target=target,
        use_bbox=strat["use_bbox"],
        target_size=strat["target_size"],
        augmentations=strat["val_augmentations"],
        train=False,
        use_dataset_mean=strat.get("use_dataset_mean", False)
    )
elif task == "verification":
    train_dataset, test_dataset, val_level = get_verification_datasets(
        image_path=IMAGE_PATH,
        base_path=BASE_PATH,
        target_size=strat["target_size"]
    )
else:
    print("Invalid task type. Exiting.")
    exit(1)

# Optionally cache the test dataset (use with caution, high memory usage)
if CACHE_TEST_DATASET:
    test_dataset = CachedDataset(test_dataset, num_workers=NUM_WORKERS)
time_dataset = time.time() - start_time
print(f"Dataset loaded in {time_dataset:.2f} seconds.")

model = strat["model"](embedding_dim=strat.get("embedding_dim"))

# Print model summary
print(f"Model: {model.__class__.__name__}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

start_time = time.time()
if task == "verification":
    if not strat.get("embedding_dim"):
        print("Error: 'embedding_dim' must be specified in the strategy for verification tasks.")
        print("Please use a strategy with 'embedding_dim' defined. (Siamese networks require this.)")
        exit(1)
    train_metrics = train_verification(
        model,
        train_dataset,
        test_dataset,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        device=DEVICE,
        use_amp=True,
        target=target,
        persistent_workers=True,
        num_workers=NUM_WORKERS,
        logger_dir=tb_log_dir,
        checkpoint_dir=checkpoint_dir,
        profiler_dir=profiler_dir,
        use_profiler=False,
        num_profiled_epochs=2,
        early_stopping_patience=EARLY_STOPPING_PATIENCE
    )
elif task == "classification":
    train_metrics = train(
        model,
        train_dataset,
        test_dataset,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        loss_type=loss_type,
        target=target,
        label_smoothing=LABEL_SMOOTHING,
        device=DEVICE,
        use_amp=True,
        persistent_workers=True,
        num_workers=NUM_WORKERS,
        logger_dir=tb_log_dir,
        checkpoint_dir=checkpoint_dir,
        profiler_dir=profiler_dir,
        use_profiler=False,
        num_profiled_epochs=2,
        early_stopping_patience=EARLY_STOPPING_PATIENCE
    )
training_time = time.time() - start_time
print(f"Training for strategy {strat['name']} completed in {training_time:.2f} seconds.")

# Save summary results
results = [{
    "strategy_key": strategy_key,
    "task": task,
    "strategy_full": serialize_strategy(strat),
    "model": model.__class__.__name__,
    "params": sum(p.numel() for p in model.parameters()),
    "num_classes": NUM_CLASSES,
    "notes": "",
    **train_metrics
}]

# Save the training summary inside the logger directory
summary_name = f"summary_{target}_{loss_type}_bs{BATCH_SIZE}_ep{NUM_EPOCHS}.pkl"
if task == "verification":
    summary_name = summary_name.replace(".pkl", f"_{val_level}.pkl")
summary_path = os.path.join(strategy_dir, summary_name)
with open(summary_path, "wb") as f:
    pickle.dump(results, f)
print(f"Training summary saved to {summary_path}")



