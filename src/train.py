import torch
import os
import gc
import time
import pickle
from torch.utils.data import DataLoader
from tqdm import tqdm
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torcheval.metrics import MulticlassAccuracy, MulticlassF1Score
from torcheval.metrics import BinaryAUROC, BinaryF1Score, BinaryPrecision, BinaryRecall

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_loss(loss_type="crossentropy", label_smoothing=0.1, alpha=1, gamma=2, num_classes=None):
    if loss_type == "crossentropy":
        return torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    elif loss_type == "focal":
        class FocalLoss(torch.nn.Module):
            def __init__(self, alpha=1, gamma=2, num_classes=None):
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma
                self.num_classes = num_classes
                
            def forward(self, inputs, targets):
                ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none')
                pt = torch.exp(-ce_loss)
                focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
                return focal_loss.mean()
        return FocalLoss(alpha=alpha, gamma=gamma, num_classes=num_classes)
    elif loss_type == "contrastive":
        class ContrastiveLoss(torch.nn.Module):
            def __init__(self, margin=1.0):
                super().__init__()
                self.margin = margin
            def forward(self, output1, output2, label):
                euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)
                loss = torch.mean(
                    label * euclidean_distance.pow(2) +
                    (1 - label) * torch.clamp(self.margin - euclidean_distance, min=0).pow(2)
                )
                return loss
        return ContrastiveLoss()
    else:
        raise ValueError(f"Loss {loss_type} not supported")

'''
Training function for PyTorch models.
This function trains a model using PyTorch.
It supports both cross-entropy and contrastive loss types.
It can also use mixed precision training with AMP.
It returns a dictionary with the best validation accuracy, balanced accuracy, best epoch, training time, and checkpoint path.
It also saves batch metrics to a pickle file.
Possibile Improvements: unified train function, as the first function does not really support contrastive loss.
'''


# --- CLASSIFICATION TRAINING FUNCTION ---
def train(
    model,
    train_dataset,
    val_dataset=None,
    batch_size=64,
    num_epochs=2,
    lr=1e-3,
    weight_decay=1e-4,
    loss_type="crossentropy",
    target="unknown",
    label_smoothing=0.1,
    checkpoint_dir="checkpoints",
    logger_dir="tb_logs",
    profiler_dir=None,
    num_workers=4,
    device=None,
    persistent_workers=True,
    use_amp=True,
    use_profiler=False,                
    num_profiled_epochs=3,
    early_stopping_patience=10          
):
    # --- SETUP & INITIALIZATION ---
    # (device, dataloaders, optimizer, scheduler, logger, metrics, etc.)
    device = device or get_device()
    model = model.to(device)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(logger_dir, exist_ok=True)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=persistent_workers
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True, persistent_workers=persistent_workers
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr * 10,  # max_lr is typically 3-10x initial lr
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,   # 30% warm-up, 70% annealing
        anneal_strategy='cos'
    )


    num_classes = train_dataset.num_classes
    criterion = get_loss(loss_type, label_smoothing, num_classes=num_classes)
    logger = TensorBoardLogger(save_dir=logger_dir, name="logs")
    version_dir = logger.log_dir
    os.makedirs(version_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        save_top_k=1,
        monitor="val_acc",
        mode="max",
        filename="best-{epoch:02d}-{val_acc:.2f}"
    )

    epochs_no_improve = 0
    early_stop = False
    best_val_acc = 0
    best_val_bal_acc = 0
    best_epoch = 0
    scaler = torch.amp.GradScaler(device.type) if use_amp and device.type == "cuda" else None


    train_acc_metric = MulticlassAccuracy(num_classes=num_classes, average="micro").to(device)
    train_bal_acc_metric = MulticlassAccuracy(num_classes=num_classes, average="macro").to(device)
    val_acc_metric = MulticlassAccuracy(num_classes=num_classes, average="micro").to(device)
    val_bal_acc_metric = MulticlassAccuracy(num_classes=num_classes, average="macro").to(device)
    train_top5_metric = MulticlassAccuracy(num_classes=num_classes, k=5).to(device)
    val_top5_metric = MulticlassAccuracy(num_classes=num_classes, k=5).to(device)
    train_f1_metric = MulticlassF1Score(num_classes=num_classes, average="macro").to(device)
    val_f1_metric = MulticlassF1Score(num_classes=num_classes, average="macro").to(device)

    train_epoch_losses = []
    train_epoch_accs = []
    val_epoch_losses = []
    val_epoch_accs = []
    val_epoch_bal_accs = []
    val_epoch_top5 = []
    val_epoch_f1 = []
    learning_rates = []

    learning_rates.append(optimizer.param_groups[0]['lr'])
    start_time = time.time()

    # ---- PROFILER PATCH ----
    if use_profiler:
        total_profiled_steps = num_profiled_epochs * (len(train_loader) + (len(val_loader) if val_loader is not None else 0))
        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=total_profiled_steps, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs', use_gzip=True),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        prof.__enter__()
        profiler_step = 0
    # ---- END PATCH ----

    # --- TRAINING LOOP ---
    for epoch in range(num_epochs):
        # --- TRAINING PHASE (one epoch) ---
        # (model.train(), loop over train_loader, optimizer step, update metrics)
        model.train()
        running_loss = 0
        total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        train_acc_metric.reset()
        train_bal_acc_metric.reset()
        train_top5_metric.reset()
        train_f1_metric.reset()
        for batch in pbar:
            optimizer.zero_grad()
            if loss_type == "crossentropy" or loss_type == "focal":
                x, y = batch
                x, y = x.to(device), y.to(device)
                valid_mask = (y >= 0) & (y < num_classes)
                if not valid_mask.any():
                    continue
                x = x[valid_mask]
                y = y[valid_mask]
                with torch.amp.autocast('cuda', enabled=use_amp and device.type == "cuda"):
                    outputs = model(x)
                    if isinstance(outputs, tuple):
                        main_output, aux_output = outputs
                        loss = criterion(main_output, y)
                        loss += 0.4 * criterion(aux_output, y)
                        outputs = main_output  
                    else:
                        loss = criterion(outputs, y)
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scheduler.step()
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                running_loss += loss.item() * x.size(0)
                preds = outputs.argmax(dim=1)
                train_acc_metric.update(preds, y)
                train_bal_acc_metric.update(preds, y)
                train_top5_metric.update(outputs, y)
                train_f1_metric.update(outputs, y)
                total += y.size(0)
                pbar.set_postfix(loss=running_loss/(total if total else 1), acc=train_acc_metric.compute().item())
            elif loss_type == "contrastive":
                x1, x2, label = batch
                x1, x2, label = x1.to(device), x2.to(device), label.to(device)
                with torch.amp.autocast('cuda', enabled=use_amp and device.type == "cuda"):
                    out1 = model(x1)
                    out2 = model(x2)
                    loss = criterion(out1, out2, label)
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scheduler.step() #step for OneCycleLR
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                    scheduler.step() #step for OneCycleLR
                running_loss += loss.item() * x1.size(0)
            # ---- PROFILER STEP ----
            if use_profiler and epoch < num_profiled_epochs:
                prof.step()
                profiler_step += 1

        # --- END OF TRAINING PHASE: compute and log epoch metrics ---    
        train_loss = running_loss / (total if total else 1)
        train_acc = train_acc_metric.compute().item()
        train_bal_acc = train_bal_acc_metric.compute().item()

        logger.experiment.add_scalar("train_loss", train_loss, epoch)
        logger.experiment.add_scalar("train_acc", train_acc, epoch)
        logger.experiment.add_scalar("train_balanced_acc", train_bal_acc, epoch)

        # --- VALIDATION PHASE (one epoch) ---
        # (model.eval(), loop over val_loader, update metrics)
        val_acc = 0
        val_loss = 0
        val_bal_acc = 0
        if val_loader is not None:
            model.eval()
            val_acc_metric.reset()
            val_bal_acc_metric.reset()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in val_loader:
                    if loss_type == "crossentropy" or loss_type == "focal":
                        x, y = batch
                        x, y = x.to(device), y.to(device)
                        valid_mask = (y >= 0) & (y < num_classes)
                        if not valid_mask.any():
                            continue
                        x = x[valid_mask]
                        y = y[valid_mask]
                        with torch.amp.autocast('cuda', enabled=use_amp and device.type == "cuda"):
                            outputs = model(x)
                            if isinstance(outputs, tuple):
                                outputs = outputs[0]
                            loss = criterion(outputs, y)
                        val_loss += loss.item() * x.size(0)
                        preds = outputs.argmax(dim=1)
                        val_acc_metric.update(preds, y)
                        val_bal_acc_metric.update(preds, y)
                        val_top5_metric.update(outputs, y)
                        val_f1_metric.update(outputs, y)
                        correct += (preds == y).sum().item()
                        total += y.size(0)
                    elif loss_type == "contrastive":
                        x1, x2, label = batch
                        x1, x2, label = x1.to(device), x2.to(device), label.to(device)
                        with torch.amp.autocast('cuda', enabled=use_amp and device.type == "cuda"):
                            out1 = model(x1)
                            out2 = model(x2)
                            loss = criterion(out1, out2, label)
                        val_loss += loss.item() * x1.size(0)
                    # ---- PROFILER STEP ----
                    if use_profiler and epoch < num_profiled_epochs:
                        prof.step()
                        profiler_step += 1
                    # ---- END PATCH ----
                val_acc = val_acc_metric.compute().item()
                val_loss = val_loss / (total if total else 1)
                val_bal_acc = val_bal_acc_metric.compute().item()
                val_top5 = val_top5_metric.compute().item()
                val_f1 = val_f1_metric.compute().item()
                # END BATCH LOOP
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Balanced Acc: {val_bal_acc:.4f}")

            # --- END OF VALIDATION PHASE: compute and log epoch metrics, early stopping, checkpoint ---
            logger.experiment.add_scalar("val_loss", val_loss, epoch)
            logger.experiment.add_scalar("val_acc", val_acc, epoch)
            logger.experiment.add_scalar("val_balanced_acc", val_bal_acc, epoch)
            logger.experiment.add_scalar("val_top5", val_top5, epoch)
            logger.experiment.add_scalar("val_f1", val_f1, epoch)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                best_checkpoint_path = os.path.join(checkpoint_dir, f"best_model_epoch{epoch+1}_acc{val_acc:.3f}_{target}_{loss_type}.pt")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1} (no improvement for {early_stopping_patience} epochs)")
                    early_stop = True
                    break

            if val_bal_acc > best_val_bal_acc:
                best_val_bal_acc = val_bal_acc
                best_epoch_bal_acc = epoch + 1
                best_bal_acc_checkpoint_path = os.path.join(checkpoint_dir, f"best_balanced_model_epoch{epoch+1}_balacc{val_bal_acc:.3f}_{target}_{loss_type}.pt")
            
            if (epoch + 1) % 25 == 0:
                torch.save(model.state_dict(), os.path.join(version_dir, f"checkpoint_epoch{epoch+1}_{target}_{loss_type}.pt"))

            # --- LOGGING & PROFILER END EPOCH ---
        
        train_epoch_losses.append(train_loss)
        train_epoch_accs.append(train_acc)
        if val_loader is not None:
            val_epoch_losses.append(val_loss)
            val_epoch_accs.append(val_acc)
            val_epoch_bal_accs.append(val_bal_acc)
            val_epoch_top5.append(val_top5)
            val_epoch_f1.append(val_f1)

        #scheduler.step(val_acc) step for ReduceLROnPlateau
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        logger.experiment.add_scalar("lr", current_lr, epoch)

        # ---- PROFILER END EPOCH ----
        if use_profiler and epoch + 1 == num_profiled_epochs:
            prof.__exit__(None, None, None)
            print(f"Profiling completed: check ./profiler_logs with TensorBoard.")

    # --- END OF TRAINING: save best checkpoints, metrics, cleanup ---
    if best_val_acc > 0:
        torch.save(model.state_dict(), best_checkpoint_path)
    if best_val_bal_acc > 0:
        torch.save(model.state_dict(), best_bal_acc_checkpoint_path)

    training_time = time.time() - start_time
    print(f"Training finished. Best val_acc: {best_val_acc:.4f} | Best val_balanced_acc: {best_val_bal_acc:.4f}")

    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # Save the best model checkpoint
    best_checkpoint_path = os.path.join(version_dir, f"{loss_type}_{target}_epoch{epoch+1}_acc{val_acc:.3f}.pt")
    torch.save(model.state_dict(), best_checkpoint_path)
    if val_bal_acc > best_val_bal_acc:
        torch.save(model.state_dict(), os.path.join(version_dir, f"{loss_type}_{target}_epoch{epoch+1}_balacc{val_bal_acc:.3f}.pt"))

    # --- RETURN SUMMARY DICT ---
    return {
        "best_val_acc": best_val_acc,
        "best_val_bal_acc": best_val_bal_acc,
        "best_epoch": best_epoch,
        "training_time": training_time,
        "checkpoint_path": best_checkpoint_path if best_val_acc > 0 else None,
        "train_loss": train_epoch_losses,
        "train_acc": train_epoch_accs,
        "val_loss": val_epoch_losses,
        "val_acc": val_epoch_accs,
        "val_bal_acc": val_epoch_bal_accs,
        "val_top5": val_epoch_top5,
        "val_f1": val_epoch_f1,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rates": learning_rates,
        "weight_decay": weight_decay,
        "loss_type": loss_type,
        "target": target,
        "label_smoothing": label_smoothing,
        "early_stopping_patience": early_stopping_patience,
        "optimizer": optimizer.__class__.__name__,
        "scheduler": scheduler.__class__.__name__
    }

# ---- VERIFICATION-SPECIFIC UTILITIES ----
def verification_accuracy(dists, labels, threshold=0.5):
    # dists, labels: torch.Tensor su GPU
    preds = (dists < threshold).float()
    acc = (preds == labels).float().mean()
    return acc.item()

import numpy as np
from sklearn.metrics import balanced_accuracy_score

def find_best_threshold(dists, labels, num_steps=100):
    # dists e labels: numpy array
    thresholds = np.linspace(dists.min(), dists.max(), num_steps)
    best_bal_acc = 0
    best_thr = 0.5
    for thr in thresholds:
        preds = (dists < thr).astype(int)
        bal_acc = balanced_accuracy_score(labels, preds)
        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_thr = thr
    return best_thr, best_bal_acc

import matplotlib.pyplot as plt
def plot_distances(dists, labels, epoch, phase, out_dir):
    plt.figure(figsize=(7,4))
    plt.hist(dists[labels==1], bins=50, alpha=0.6, label="Positive (label=1)")
    plt.hist(dists[labels==0], bins=50, alpha=0.6, label="Negative (label=0)")
    plt.xlabel("Distance")
    plt.ylabel("Count")
    plt.title(f"Distribution of Pairwise Distances ({phase}) - Epoch {epoch+1}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/distances_{phase}_epoch{epoch+1}.png")
    plt.close()


def train_verification(
    model,
    train_dataset,
    val_dataset=None,
    batch_size=64,
    num_epochs=2,
    lr=1e-3,
    weight_decay=1e-4,
    loss_type="contrastive",
    checkpoint_dir="checkpoints",
    target="unknown",
    logger_dir="tb_logs",
    profiler_dir=None,
    num_workers=4,
    device=None,
    persistent_workers=True,
    use_amp=True,
    use_profiler=False,                
    num_profiled_epochs=3,
    early_stopping_patience=10   
):
    device = device or get_device()
    model = model.to(device)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(logger_dir, exist_ok=True)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=persistent_workers
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True, persistent_workers=persistent_workers
        )

    criterion = get_loss(loss_type)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = len(train_loader)
    '''
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr * 3,           
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.5,           
        anneal_strategy='cos',
        div_factor=25,
        final_div_factor=10
    )
    '''
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
    )
    logger = TensorBoardLogger(save_dir=logger_dir, name="logs")
    version_dir = logger.log_dir
    os.makedirs(version_dir, exist_ok=True)
    scaler = torch.amp.GradScaler(device.type) if use_amp and device.type == "cuda" else None

    train_epoch_losses = []
    train_accs = []
    train_dists = []
    train_labels = []   
    val_epoch_losses = []
    val_epoch_acc = []
    val_epoch_bal_acc = []
    val_epoch_roc_auc = []
    val_epoch_prec = []
    val_epoch_rec = []
    val_epoch_f1 = []
    learning_rates = []

    learning_rates.append(optimizer.param_groups[0]['lr'])

    best_val_loss = float("inf")
    best_epoch = 0
    best_checkpoint_path = ""
    epochs_no_improve = 0
    early_stop = False

    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            optimizer.zero_grad()
            x1, x2, label = batch
            x1, x2, label = x1.to(device).float(), x2.to(device).float(), label.to(device).float()
            with torch.amp.autocast('cuda', enabled=use_amp and device.type == "cuda"):
                out1, out2 = model(x1, x2)
                loss = criterion(out1, out2, label)
                dist = torch.nn.functional.pairwise_distance(out1, out2)
                train_dists.append(dist.detach().cpu())
                train_labels.append(label.detach().cpu())
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                #scheduler.step() # OneCycleLR
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
                #scheduler.step() #OneCycleLR
            running_loss += loss.item() * x1.size(0)
            total += x1.size(0)
            # Calculate batch accuracy
            batch_acc = verification_accuracy(dist, label, threshold=0.5)
            pbar.set_postfix(loss=running_loss/(total if total else 1), acc=batch_acc)

        train_loss = running_loss / (total if total else 1)
        logger.experiment.add_scalar("train_loss", train_loss, epoch)
        train_epoch_losses.append(train_loss)

        if len(train_dists) > 0:
            train_dists = torch.cat(train_dists)
            train_labels = torch.cat(train_labels)
            #t_dists_np = train_dists.cpu().numpy()
            #t_labels_np = train_labels.cpu().numpy()
            train_acc = verification_accuracy(train_dists, train_labels, threshold=0.5)
            #print(f"Train Acc: {train_acc:.4f}")
            #plot_distances(t_dists_np, t_labels_np, epoch, "train", checkpoint_dir)
            train_accs.append(train_acc)
            train_dists = []
            train_labels = []

        # --- VALIDATION PHASE ---
        val_loss = 0
        val_total = 0
        all_dists = []
        all_labels = []
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    if loss_type == "contrastive":
                        x1, x2, label = batch
                        x1, x2, label = x1.to(device).float(), x2.to(device).float(), label.to(device).float()
                        with torch.amp.autocast('cuda', enabled=use_amp and device.type == "cuda"):
                            out1, out2 = model(x1, x2)
                            loss = criterion(out1, out2, label)
                        batch_size_actual = x1.size(0)
                        # Distances for accuracy computation
                        dist = torch.nn.functional.pairwise_distance(out1, out2)
                        all_dists.append(dist)
                        all_labels.append(label)
                    val_loss += loss.item() * batch_size_actual
                    val_total += batch_size_actual
            val_loss = val_loss / (val_total if val_total else 1)
            logger.experiment.add_scalar("val_loss", val_loss, epoch)
            val_epoch_losses.append(val_loss)
            scheduler.step(val_loss) # ReduceLROnPlateau

            # --- ACCURACIES---
            all_dists = torch.cat(all_dists)
            all_labels = torch.cat(all_labels)
            

            dists_np = all_dists.cpu().numpy()
            labels_np = all_labels.cpu().numpy()
            threshold, val_bal_acc = find_best_threshold(dists_np, labels_np)
            #plot_distances(dists_np, labels_np, epoch, "val", checkpoint_dir)

            preds = (all_dists < threshold).int()
            all_labels_int = all_labels.int()

            val_acc = verification_accuracy(all_dists, all_labels, threshold)

            # ROC-AUC
            roc_auc_metric = BinaryAUROC().to(all_dists.device)
            roc_auc_metric.update(-all_dists, all_labels)  # -dists: più piccolo = più simile
            val_roc_auc = roc_auc_metric.compute().item()

            # F1
            f1_metric = BinaryF1Score().to(all_dists.device)
            f1_metric.update(preds, all_labels_int)
            val_f1 = f1_metric.compute().item()

            # Precision
            precision_metric = BinaryPrecision().to(all_dists.device)
            precision_metric.update(preds, all_labels_int)
            val_precision = precision_metric.compute().item()

            # Recall
            recall_metric = BinaryRecall().to(all_dists.device)
            recall_metric.update(preds, all_labels_int)
            val_recall = recall_metric.compute().item()

            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Balanced Acc: {val_bal_acc:.4f}")
            print(f"Val ROC-AUC: {val_roc_auc:.4f} | F1: {val_f1:.4f} | Precision: {val_precision:.4f} | Recall: {val_recall:.4f}")
            logger.experiment.add_scalar("val_roc_auc", val_roc_auc, epoch)
            logger.experiment.add_scalar("val_f1", val_f1, epoch)
            logger.experiment.add_scalar("val_precision", val_precision, epoch)
            logger.experiment.add_scalar("val_recall", val_recall, epoch)
            logger.experiment.add_scalar("val_acc", val_acc, epoch)
            logger.experiment.add_scalar("val_balanced_acc", val_bal_acc, epoch)

            val_epoch_acc.append(val_acc)
            val_epoch_bal_acc.append(val_bal_acc)
            val_epoch_roc_auc.append(val_roc_auc)
            val_epoch_f1.append(val_f1)
            val_epoch_prec.append(val_precision)
            val_epoch_rec.append(val_recall)
            

            
            # --- EARLY STOPPING & CHECKPOINT ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                best_checkpoint_path = os.path.join(checkpoint_dir, f"best_model_epoch{epoch+1}_val_loss{val_loss:.4f}_{target}_{loss_type}.pt")
                #torch.save(model.state_dict(), best_checkpoint_path)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1} (no improvement for {early_stopping_patience} epochs)")
                    early_stop = True
                    break

            if (epoch + 1) % 25 == 0:
                torch.save(model.state_dict(), os.path.join(version_dir, f"checkpoint_epoch{epoch+1}_{target}_{loss_type}.pt"))

        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        logger.experiment.add_scalar("lr", current_lr, epoch)

        if early_stop:
            break

    training_time = time.time() - start_time
    print(f"Training finished. Best val_loss: {best_val_loss:.4f}")

    if device.type == "cuda":
        torch.cuda.empty_cache()
    import gc
    gc.collect()

    # --- RETURN SUMMARY DICT ---
    return {
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "training_time": training_time,
        "checkpoint_path": best_checkpoint_path,
        "train_loss": train_epoch_losses,
        "train_acc": train_accs,
        "val_loss": val_epoch_losses,
        "val_acc": val_epoch_acc if val_loader is not None else None,
        "val_bal_acc": val_epoch_bal_acc if val_loader is not None else None,
        "val_roc_auc": val_epoch_roc_auc if val_loader is not None else None,
        "val_f1": val_epoch_f1 if val_loader is not None else None,
        "val_precision": val_epoch_prec if val_loader is not None else None,
        "val_recall": val_epoch_rec if val_loader is not None else None,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rates": learning_rates,
        "weight_decay": weight_decay,
        "loss_type": loss_type,
        "early_stopping_patience": early_stopping_patience,
        "optimizer": optimizer.__class__.__name__,
        "scheduler": scheduler.__class__.__name__,
        "all_dists": dists_np if val_loader is not None else None, # for ROC curve
        "all_labels": labels_np if val_loader is not None else None # for ROC curve
    }
