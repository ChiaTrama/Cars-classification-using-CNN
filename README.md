## Efficient Fine-Grained Vehicle Recognition under Class Imbalance  
Fine-grained vehicle **classification** and **verification** on the [CompCars dataset](http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/) using Convolutional Neural Networks (PyTorch).  
Developed for the *Neural Networks and Deep Learning* course — MSc Physics of Data, University of Padova (A.Y. 2024–2025).

---

### Authors  
**Chiara Tramarin**  
**Alessio Tuscano**

---

### Project Description  
This project analyzes how different CNN architectures perform under **severe class imbalance** in fine-grained vehicle recognition.  
We evaluate both **classification** (make/model prediction) and **verification** (Siamese similarity) tasks using **Cross-Entropy**, **Focal Loss**, and **Contrastive Loss**.

Several pretrained and custom models are compared, including **InceptionV3**, **ResNet18**, **EfficientNet-B0**, and custom **SimpleResNet** variants trained from scratch.  
Each configuration is evaluated in terms of **Balanced Accuracy**, **F1-score**, and **training efficiency**.

Results show that lightweight pretrained models such as *ResNet18* provide the best trade-off between accuracy and efficiency, while *InceptionV3* achieves the highest balanced accuracy overall.  
For verification, a *Siamese ResNet18* backbone learns robust embeddings and maintains strong **ROC–AUC** across all difficulty levels.

The codebase is fully modular, using **PyTorch**, **Albumentations**, and **TensorBoard** for logging and analysis.

---

### Dataset  
All experiments are based on the **CompCars** dataset (Yang et al., CVPR 2015),  
using the official train/test splits for classification and the `verification_pairs_*` lists for the verification task.

---

### Repository Structure  
- **main.py** — interactive CLI entry point: select *task* (classification/verification), *target* (make/model), *loss*, and *strategy*.  
  Loads dataset/model, runs training, and saves metric summaries (`.pkl`) under the strategy folder.  
  Supports dataset caching and checkpoint management.    
- **train.py** (root) — pure PyTorch training loop for both tasks:  
  - *Classification*: Cross-Entropy / Focal Loss, OneCycleLR scheduler, metrics (**Accuracy**, **Balanced Accuracy**, **Top-5**, **F1**), early stopping, TensorBoard logging.  
  - *Verification*: Contrastive Loss, automatic threshold optimization, metrics (**ROC–AUC**, **F1**, **Precision**, **Recall**).    

- **src/**
  - **dataset.py** — CompCars dataset loader for classification:  
    - builds global *make/model → class id* mapping from `train.txt`/`test.txt`;  
    - optional **use_bbox** cropping;  
    - **Albumentations** augmentations and normalization (ImageNet or dataset-specific mean/std);  
    - **CachedDataset** for full in-memory caching. :contentReference[oaicite:0]{index=0}  
  - **dataset_verification.py** — Siamese dataset logic for verification:  
    - `CompCarsVerificationDataset` reads pre-built pairs (`easy`, `medium`, `hard`);  
    - `CompCarsBaseDataset` + `SiameseDataset` dynamically generate positive/negative pairs. :contentReference[oaicite:1]{index=1}  
  - **model.py** — CNN architectures:  
    - **SimpleResNet** / **SimpleResNetLarge** (trained from scratch);  
    - **ResNet18** and **InceptionV3** (fine-tuning, auxiliary head for Inception);  
    - **SiameseResNet18**, **SiameseResNet50_30M**, and **SiameseEfficientNet-B0** with BN-neck and L2-normalized embeddings. :contentReference[oaicite:2]{index=2}  
  - **strategies.py** — predefined model/training strategies:  
    - specifies input size, augmentations, and pretrained weights;  
    - examples: `InceptionV3_299x299`, `ResNet18_FineTuning_224x224`, `SimpleResNet*`, `SiameseResNet18_192x192`, etc.;  
    - includes `serialize_strategy(...)` for structured logging. :contentReference[oaicite:3]{index=3}  
  - **train.py** — internal modular training functions imported by `main.py`; implements the same metric logic and checkpoint saving for classification and verification. :contentReference[oaicite:4]{index=4}  
  - **lightning_train.py** — legacy **PyTorch Lightning** version of the training loop (kept for compatibility). :contentReference[oaicite:5]{index=5}  

- **runs/** — automatically created output directory containing:
  - checkpoints, TensorBoard logs, profiler traces, and `.pkl` metric summaries.  
- **NNDL_project_ChiaraTramarin_AlessioTuscano.pdf** — final report (paper) describing the full experimental study.  

---

### Quick Start  
```bash
pip install -r requirements.txt
python main.py
# Select: task, target, loss, strategy, epochs, batch size
# Outputs saved under: runs/<strategy>/
