import os
import json
import cv2
import torch
from torch.utils.data import Dataset
from config import SCALE
class VideoFramesDataset(Dataset):
    """
        Класс для работы с кадрами видео и аннотациями в формате COCO.

        Args:
            frames_dir: (str) Путь к директории с кадрами видео.
            annotation_file (str): Путь к COCO-аннотациям (.json).
            transform (callable, optional): Преобразования для изображения (ToTensor).

        Returns:
            tuple: (img, target)
                img (torch.Tensor или np.array): Изображение после transform.
                target (dict): Словарь с ключами:
                    - 'boxes' (Tensor[N, 4]): координаты bbox.
                    - 'labels' (Tensor[N]): категории объектов.
                    - 'image_id' (Tensor[1]): id кадра.
                    - 'image_name' (str): имя файла кадра.
        """
    def __init__(self, frames_dir, annotation_file, transform=None):
        self.frames_dir = frames_dir
        self.transform = transform
        with open(annotation_file, "r") as f:
            self.coco = json.load(f)

        self.images = {im['id']: im for im in self.coco['images']}
        self.annotations = {}
        for ann in self.coco['annotations']:
            self.annotations.setdefault(ann['image_id'], []).append(ann)

        self.ids = list(self.images.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.images[img_id]
        img_path = os.path.join(self.frames_dir, img_info['file_name'])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if SCALE != 1.0:
            height, width = int(img.shape[0]*SCALE), int(img.shape[1]*SCALE)
            img = cv2.resize(img, (width, height))

        boxes = []
        labels = []
        for ann in self.annotations.get(img_id, []):
            x, y, w, h = ann['bbox']
            if SCALE != 1.0:
                x, y, w, h = x * SCALE, y * SCALE, w * SCALE, h * SCALE
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])


        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        target = {"boxes": boxes,
                  "labels": labels,
                  "image_id": torch.tensor([img_id]),
                  "image_name": img_info['file_name']
                }

        if self.transform:
            img = self.transform(img)

        return img, target