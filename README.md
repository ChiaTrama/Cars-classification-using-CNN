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
- **main.py** — interactive CLI: choose task (classification / verification), target (make / model), loss, and strategy; builds datasets/models, runs training, and saves metric summaries (`.pkl`), checkpoints, and logs. 

- **src/**
  - `dataset.py` — CompCars classification dataset: global make/model→id mapping from the official splits; optional bbox cropping; Albumentations augmentations; ImageNet or dataset mean/std normalization; optional in-RAM caching via `CachedDataset`. 
  - `dataset_verification.py` — verification datasets for Siamese training/validation: reads **predefined pair lists** (`easy`, `medium`, `hard`) for val, and uses `CompCarsBaseDataset` + `SiameseDataset` to build **on-the-fly positive/negative pairs** for training. 
  - `model.py` — CNN and Siamese architectures: SimpleResNet / SimpleResNetLarge (from scratch); ResNet18 & InceptionV3 for fine-tuning (with aux head for Inception); Siamese backbones (ResNet18, ResNet50_30M, EfficientNet-B0) with BN-neck and L2-normalized embeddings. 
  - `strategies.py` — predefined training “strategies” (input size, augmentations, bbox usage, normalization choice, model factory) for both classification and verification; includes `serialize_strategy(...)` for saving metadata.
  - `train.py` — **core PyTorch training loop** used by `main.py`.  
    - *Classification:* Cross-Entropy / Focal Loss, OneCycleLR, metrics (**Accuracy**, **Balanced Accuracy**, **Top-5**, **F1**), early stopping, TensorBoard logging.  
    - *Verification:* Contrastive loss, automatic threshold search, metrics (**ROC-AUC**, **F1**, **Precision**, **Recall**). 
  - `lightning_train.py` — legacy PyTorch Lightning training module kept for compatibility.

- **runs/** — auto-generated outputs per strategy: checkpoints, TensorBoard logs, profiler traces, and `.pkl` summaries. (Created by `main.py`.) 
- **NNDL_project_ChiaraTramarin_AlessioTuscano.pdf** — final report (paper) with results and analysis. 


---


