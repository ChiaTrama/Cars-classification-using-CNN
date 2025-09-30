import os
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from joblib import Parallel, delayed

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# Default ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class CachedDataset(torch.utils.data.Dataset):
    """Caches all samples of a dataset in memory for faster access."""
    def __init__(self, base_dataset, num_workers=None):
        self.base_dataset = base_dataset
        self.num_workers = num_workers or os.cpu_count()
        self.cache = Parallel(n_jobs=self.num_workers)(
            delayed(lambda idx: self.base_dataset[idx])(idx) for idx in range(len(self.base_dataset))
        )

    def __len__(self):
        return len(self.cache)
    def __getitem__(self, idx):
        return self.cache[idx]


def crop_with_bbox(img, bbox, padding=0.1):
    """Crop the image using the bounding box with optional padding."""
    if bbox is None:
        return img
    x1, y1, x2, y2 = bbox
    w, h = img.size
    pad_w = int((x2 - x1) * padding)
    pad_h = int((y2 - y1) * padding)
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(w, x2 + pad_w)
    y2 = min(h, y2 + pad_h)
    return img.crop((x1, y1, x2, y2))

def get_albu_transform(
    target_size=(224, 224),
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
    train=True
):
    """Return an albumentations Compose transform for train or val/test."""
    if train:
        return A.Compose([
            A.Resize(height=target_size[1], width=target_size[0]),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Affine(
                scale=(0.9, 1.1),  # oppure (0.95, 1.05) per meno variazione
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                rotate=(-10, 10),
                shear={"x": (-5, 5), "y": (-2, 2)},
                interpolation=cv2.INTER_LINEAR,
                fit_output=False,
                p=0.3
            ),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(height=target_size[1], width=target_size[0]),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])

class CompCarsDataset(Dataset):
    """Dataset for the CompCars dataset.

    This dataset supports both 'make' and 'model' targets.
    It can use bounding boxes for cropping images.
    It supports custom augmentations via albumentations.
    """
    # Class variables for global mapping
    make_id_to_class = None
    model_id_to_class = None

    def __init__(
        self,
        split_file,
        image_dir,
        label_dir,
        target='make',  # 'make' or 'model'
        use_bbox=False,
        target_size=(224, 224),
        use_dataset_mean=False,
        train=True,  # True: train augmentations, False: val/test
        padding=0.1,
        augmentations=None  # Optional: custom albumentations pipeline
    ):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.split_file = split_file
        self.target = target
        self.use_bbox = use_bbox
        self.target_size = target_size
        self.padding = padding

        # Use provided mean/std or default to ImageNet
        ''' these values were computed on the training set:
        Dataset mean: [0.4913326824655994, 0.47955603344650855, 0.46958770148374746]
        Dataset std: [0.2585349967525874, 0.25819146443271296, 0.2620055419357117]
        '''
        DATASET_MEAN = [0.4913326824655994, 0.47955603344650855, 0.46958770148374746]
        DATASET_STD = [0.2585349967525874, 0.25819146443271296, 0.2620055419357117]

        if use_dataset_mean:
            self.mean = DATASET_MEAN
            self.std = DATASET_STD
        else:
            self.mean = IMAGENET_MEAN
            self.std = IMAGENET_STD

        # Albumentations pipeline
        if augmentations is not None:
            self.augmentations = augmentations
        else:
            self.augmentations = get_albu_transform(
                target_size=self.target_size,
                mean=self.mean,
                std=self.std,
                train=train
            )

        self.samples = self._load_samples(split_file)
        self._ensure_global_mapping()
        self._assign_class_labels()

    def _parse_path(self, path):
        # Typical path: "78/1/2010/439374a1456969.jpg"
        parts = path.strip().split('/')
        if len(parts) >= 4:
            make_id = int(parts[0])
            model_id = int(parts[1])
            year = parts[2]
            return make_id, model_id, year
        return -1, -1, "unknown"

    def _load_samples(self, split_file):
        """Load all samples from the split file."""
        samples = []
        with open(split_file, 'r') as f:
            for line in f:
                path = line.strip()
                make_id, model_id, year = self._parse_path(path)
                bbox = self._load_bbox(path)
                samples.append({
                    'img_path': os.path.join(self.image_dir, path),
                    'bbox': bbox,
                    'make_id': make_id,
                    'model_id': model_id
                })
        return samples

    def _load_bbox(self, path):
        """Load bounding box coordinates from the label file."""
        label_path = os.path.join(self.label_dir, path.replace('.jpg', '.txt'))
        if not os.path.exists(label_path):
            return None
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
                if len(lines) >= 3:
                    coords = list(map(int, lines[2].strip().split()))
                    if len(coords) == 4:
                        return coords
        except Exception:
            pass
        return None

    def _ensure_global_mapping(self):
        """
        Build the global mapping from all make_id/model_id present in both train.txt and test.txt
        in the same directory as the current split_file.
        """
        if (self.target == 'make' and CompCarsDataset.make_id_to_class is not None) or \
           (self.target == 'model' and CompCarsDataset.model_id_to_class is not None):
            return

        split_dir = os.path.dirname(self.split_file)
        ids = set()
        for split in ["train.txt", "test.txt"]:
            split_path = os.path.join(split_dir, split)
            if not os.path.exists(split_path):
                continue
            with open(split_path) as f:
                for line in f:
                    make_id, model_id, _ = self._parse_path(line.strip())
                    if self.target == 'make':
                        ids.add(make_id)
                    elif self.target == 'model':
                        ids.add(model_id)
        id_to_class = {id_: idx for idx, id_ in enumerate(sorted(ids))}
        if self.target == 'make':
            CompCarsDataset.make_id_to_class = id_to_class
        elif self.target == 'model':
            CompCarsDataset.model_id_to_class = id_to_class
        else:
            raise ValueError(f"Target {self.target} not supported (use 'make' or 'model')")

    def _assign_class_labels(self):
        """Assign the correct class label to each sample."""
        if self.target == 'make':
            id_to_class = CompCarsDataset.make_id_to_class
            for s in self.samples:
                s['class_label'] = id_to_class[s['make_id']]
            self.num_classes = len(id_to_class)
        elif self.target == 'model':
            id_to_class = CompCarsDataset.model_id_to_class
            for s in self.samples:
                s['class_label'] = id_to_class[s['model_id']]
            self.num_classes = len(id_to_class)
        else:
            raise ValueError(f"Target {self.target} not supported (use 'make' or 'model')")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            img = Image.open(sample['img_path']).convert('RGB')
        except Exception as e:
            print(f"Warning: corrupted image {sample['img_path']}: {e}")
            img = Image.new('RGB', self.target_size)
        if self.use_bbox and sample['bbox']:
            img = crop_with_bbox(img, sample['bbox'], self.padding)
        img = img.resize(self.target_size)
        img = np.array(img)
        if self.augmentations:
            img = self.augmentations(image=img)['image']
        target = sample['class_label']
        return img, target