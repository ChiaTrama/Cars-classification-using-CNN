import os
from PIL import Image
import numpy as np
import random
import torch
from torch.utils.data import Dataset

class CompCarsVerificationDataset(Dataset):
    """
    Dataset for vehicle verification tasks using the CompCars dataset.
    Each item is a tuple of two images and a label indicating if they are of the same vehicle.
    """
    def __init__(
        self,
        pairs_file,
        image_dir,
        target_size=(224, 224),
        augmentations=None,
        train=True
    ):
        self.image_dir = image_dir
        self.target_size = target_size
        self.augmentations = augmentations
        self.train = train
        self.pairs = self._load_pairs(pairs_file)

    def _load_pairs(self, pairs_file):
        pairs = []
        with open(pairs_file, 'r') as f:
            for line in f:
                p1, p2, label = line.strip().split()
                pairs.append((p1, p2, int(label)))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def _load_image(self, rel_path):
        img_path = os.path.join(self.image_dir, rel_path)
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Warning: corrupted image {img_path}: {e}")
            img = Image.new('RGB', self.target_size)
        img = img.resize(self.target_size)
        img = np.array(img)
        if img.shape != (self.target_size[0], self.target_size[1], 3):
            print(f"Warning: image {img_path} has wrong shape {img.shape}, expected {(self.target_size[0], self.target_size[1], 3)}")
            img = np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.uint8)
        return img

    def __getitem__(self, idx):
        p1, p2, label = self.pairs[idx]
        img1 = self._load_image(p1)
        img2 = self._load_image(p2)
        if self.augmentations:
            img1 = self.augmentations(image=img1)['image']
            img2 = self.augmentations(image=img2)['image']
        return img1, img2, torch.tensor(label, dtype=torch.float32)


class CompCarsBaseDataset(Dataset):
    def __init__(self, image_list_file, label_dir, image_dir, target="model", target_size=(224, 224), augmentations=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.target_size = target_size
        self.augmentations = augmentations
        self.target = target
        self.img_paths = []
        self.labels = []
        with open(image_list_file, "r") as f:
            for line in f:
                rel_path = line.strip()
                if rel_path:
                    self.img_paths.append(rel_path)
                    label = self._get_label(rel_path)
                    self.labels.append(label)

    def _get_label(self, rel_path):
        if self.target == "make":
            return rel_path.split("/")[0]
        elif self.target == "model":
            return rel_path.split("/")[1]
        else:
            raise ValueError("Target unsupported")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.img_paths[idx])
        img = Image.open(img_path).convert('RGB').resize(self.target_size)
        img = np.array(img)
        if self.augmentations:
            img = self.augmentations(image=img)['image']
        return {'IMAGE': img, 'LABEL': self.labels[idx]}


class SiameseDataset(Dataset):
    """
    Dataset for training Siamese networks.
    Each item is a tuple of two images and a label indicating if they are of the same class.
    Generates pairs on-the-fly.
    """
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.labels_to_indices = {}

        # Build a mapping from labels to indices
        for idx in range(len(self.base_dataset)):
            label = self.base_dataset[idx]['LABEL']
            if label not in self.labels_to_indices:
                self.labels_to_indices[label] = []
            self.labels_to_indices[label].append(idx)
        self.labels = list(self.labels_to_indices.keys())

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        sample1 = self.base_dataset[index]
        label1 = sample1['LABEL']
        img1 = sample1['IMAGE']

        should_get_same_class = random.randint(0, 1)

        if should_get_same_class:
            idx2 = index
            while idx2 == index:
                idx2 = random.choice(self.labels_to_indices[label1])
        else:
            label2 = random.choice([l for l in self.labels if l != label1])
            idx2 = random.choice(self.labels_to_indices[label2])

        sample2 = self.base_dataset[idx2]
        img2 = sample2['IMAGE']
        label = 1 if should_get_same_class else 0

        return img1, img2, torch.tensor(label, dtype=torch.float32)