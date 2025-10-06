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
Each configuration is evaluated through multiple metrics, including Accuracy, Balanced Accuracy, F1-score, Top-5 accuracy, and training efficiency.

Results show that lightweight pretrained models such as *ResNet18* provide the best trade-off between accuracy and efficiency, while *InceptionV3* achieves the highest balanced accuracy overall.  
For verification, a *Siamese ResNet18* backbone learns robust embeddings and maintains strong **ROC–AUC** across all difficulty levels.

The codebase is fully modular, using **PyTorch**, **Albumentations**, and **TensorBoard** for logging and analysis.

---

### Dataset  
All experiments are based on the **CompCars** dataset (Yang et al., CVPR 2015),  
using the official train/test splits for classification and the verification pairs lists for the verification task.

---

### Repository Structure  
- **main.py** — main interactive script: selects task (classification / verification), target (make / model), loss, and strategy.  
  Loads datasets and models, runs training, and saves `.pkl` summaries, checkpoints, and logs.  

- **src/**
  - `dataset.py` — classification dataset loader (CompCars), with Albumentations augmentations, normalization (ImageNet or dataset mean), and optional caching.  
  - `dataset_verification.py` — Siamese verification datasets; handles predefined pair lists (easy / medium / hard) or dynamic pair generation.  
  - `model.py` — CNN and Siamese architectures: *SimpleResNet*, *ResNet18*, *InceptionV3*, *EfficientNet-B0*, etc.  
  - `strategies.py` — predefined model/training setups (input size, augmentations, pretrained weights); used to standardize experiments.  
  - `train.py` — core PyTorch training loop for both tasks:  
    - *Classification*: Cross-Entropy / Focal Loss, OneCycleLR, metrics (**Acc**, **BalAcc**, **Top-5**, **F1**), early stopping, TensorBoard logging.  
    - *Verification*: Contrastive Loss, optimal threshold search, metrics (**ROC–AUC**, **F1**, **Precision**, **Recall**).  
  - `lightning_train.py` — legacy implementation using PyTorch Lightning (kept for compatibility).  

- **runs/** — automatically created output folder containing checkpoints, TensorBoard logs, profiler traces, and `.pkl` summaries.  
- **NNDL_project_ChiaraTramarin_AlessioTuscano.pdf** — final report (paper) with results and analysis.

---


