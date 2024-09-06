import os
import torch
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset


class PollenDataset(Dataset):
    def __init__(self, csv_file, root_dir, transforms=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transforms = transforms
        self.class_map = {
            'buckwheat': 0, 'clover': 1, 'angelica': 2, 'angelica_garden': 3, 'willow': 4,
            'hill_mustard': 5, 'linden': 6, 'meadow_pink': 7, 'alder': 8, 'birch': 9, 'fireweed': 10,
            'nettle': 11, 'pigweed': 12, 'plantain': 13, 'sorrel': 14, 'grass': 15, 'pine': 16,
            'maple': 17, 'hazel': 18, 'mugwort': 19
        }

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        xmin = self.annotations.iloc[idx, 1]
        ymin = self.annotations.iloc[idx, 2]
        xmax = self.annotations.iloc[idx, 3]
        ymax = self.annotations.iloc[idx, 4]
        label = self.class_map[self.annotations.iloc[idx, 5]]

        # Формируем словарь с координатами бокса и классом
        boxes = torch.tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32)
        labels = torch.tensor([label], dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}

        if self.transforms:
            image = self.transforms(image)

        return image, target
