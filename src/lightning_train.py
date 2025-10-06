import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer, LightningModule
from sklearn.metrics import balanced_accuracy_score
from torch.optim.lr_scheduler import StepLR
import gc

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_loss(loss_type="crossentropy"):
    if loss_type == "crossentropy":
        return torch.nn.CrossEntropyLoss()
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
Old interface for training, kept for compatibility and checking purposes.
This function trains a model using PyTorch Lightning.
It supports both cross-entropy and contrastive loss types.
It can also use mixed precision training with AMP.
'''

def train(
    model,
    train_dataset,
    val_dataset=None,
    batch_size=64,
    num_epochs=10,
    lr=1e-3,
    loss_type="crossentropy",
    checkpoint_dir="checkpoints",
    logger_dir="tb_logs",
    num_workers=4,
    metrics=["accuracy", "balanced_accuracy"],
    device=None,
    persistent_workers=True,
    use_amp=False,
    use_lightning_trainer=True
):
    device = device or get_device()
    #if not use_lightning_trainer:
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

    class LitModule(LightningModule):
        def __init__(self, model, lr, loss_type):
            super().__init__()
            self.model = model
            self.lr = lr
            self.loss_type = loss_type
            self.criterion = get_loss(loss_type)
        def forward(self, x):
            return self.model(x)
        def training_step(self, batch, batch_idx):
            if self.loss_type == "crossentropy":
                x, y = batch
                logits = self(x)
                loss = self.criterion(logits, y)
                acc = (logits.argmax(dim=1) == y).float().mean()
                self.log('train_loss', loss, on_step=False, on_epoch=True)
                self.log('train_acc', acc, on_step=False, on_epoch=True)
                return loss
            elif self.loss_type == "contrastive":
                x1, x2, label = batch
                out1 = self(x1)
                out2 = self(x2)
                loss = self.criterion(out1, out2, label)
                self.log('train_loss', loss, on_step=False, on_epoch=True)
                return loss
        def validation_step(self, batch, batch_idx):
            if self.loss_type == "crossentropy":
                x, y = batch
                logits = self(x)
                loss = self.criterion(logits, y)
                acc = (logits.argmax(dim=1) == y).float().mean()
                self.log('val_loss', loss, on_step=False, on_epoch=True)
                self.log('val_acc', acc, on_step=False, on_epoch=True)
            elif self.loss_type == "contrastive":
                x1, x2, label = batch
                out1 = self(x1)
                out2 = self(x2)
                loss = self.criterion(out1, out2, label)
                self.log('val_loss', loss, on_step=False, on_epoch=True)
        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            return optimizer

    lightning_model = LitModule(model, lr, loss_type)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        save_top_k=1,
        monitor="val_acc",
        mode="max",
        filename="best-{epoch:02d}-{val_acc:.2f}"
    )
    logger = TensorBoardLogger(save_dir=logger_dir, name="compcars")
    trainer = Trainer(
        max_epochs=num_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=16-mixed if use_amp else 32,
        callbacks=[checkpoint_callback],
        logger=logger,
        log_every_n_steps=10,
        deterministic=True
    )
    trainer.fit(lightning_model, train_loader, val_loader)
    if device.type == "cuda":
        lightning_model.to(torch.device("cpu"))
        torch.cuda.empty_cache()
        gc.collect()  # Ensure model is on the correct device
    return lightning_model

# Example usage:
# trained_model = train(
#     model, train_dataset, val_dataset,
#     loss_type="crossentropy",
#     use_amp=True,
#     use_lightning_trainer=False
# )
