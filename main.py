"""
Person Detection in Video using Faster R-CNN.

Usage:
    python main.py --mode train
    python main.py --mode infer
"""

import argparse
import json
from config import *
from data.video_dataset import VideoFramesDataset
from model.detector import get_model, train_model
from inference.video_processor import process_video
from torch.utils.data import DataLoader, random_split
from torchvision import transforms as T

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "infer"], required=True)
    args = parser.parse_args()
    transform = T.Compose([T.ToTensor()])
    if args.mode == "train":
        dataset = VideoFramesDataset(INPUT_FRAMES_DIR, ANNOTATION_FILE, transform=transform)
        train_size = int(0.7 * len(dataset))
        val_size = len(dataset) - train_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  collate_fn=lambda x: tuple(zip(*x)))
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                collate_fn=lambda x: tuple(zip(*x)))

        model = get_model()
        train_model(model, train_loader, NUM_EPOCHS)
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Модель сохранена в {MODEL_PATH}")

    elif args.mode == "infer":
        model = get_model(pretrained=False)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()

        results, avg_time = process_video(model, transform=transform)
        with open("results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"Видео сохранено: {OUTPUT_VIDEO}, среднее время: {avg_time:.3f}с")

if __name__ == "__main__":
    main()