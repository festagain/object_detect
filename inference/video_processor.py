import cv2
import torch
import time
import json
from pathlib import Path
import torchvision.transforms
from config import *
from model.utils import draw_transparent_box, non_max_suppression

def process_video(model, transform):
    """Прогоняет модель по видео, выполняет детекцию объектов (person) и сохраняет результат в видео с
    наложенными bounding box'ами.

    Args:
        model (torch.nn.Module): Предобученная модель
    Returns:
        tuple:
            coco_boxes (list of dict): Список словарей в формате COCO для каждого обнаруженного объекта.
                    - 'image_id' (int): индекс кадра в видео.
                    - 'category_id' (int): ID класса (например, PERSON_CLASS_ID).
                    - 'bbox' (list of float): [x, y, width, height] ограничивающий прямоугольник.
                    - 'score' (float): уверенность модели для данного бокса.
            avg_time_per_frame (float): Среднее время инференса на один кадр в секундах.
    """

    model.eval()
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео {INPUT_VIDEO}")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * SCALE)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * SCALE)
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps / DOWNSAMPLE_FOR_SPEED, (width, height))
    frame_idx = 0
    total_time = 0.0
    processed = 0

    coco_boxes = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            if SCALE != 1.0:
                frame = cv2.resize(frame, (width, height))

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = transform(img_rgb).to(DEVICE)

            t0 = time.time()
            with torch.no_grad():
                outputs = model([tensor])
            t1 = time.time()
            elapsed = t1 - t0
            total_time += elapsed
            processed += 1

            out_dict = outputs[0]
            boxes = out_dict['boxes'].cpu().numpy()
            labels = out_dict['labels'].cpu().numpy()
            scores = out_dict['scores'].cpu().numpy()

            mask_person = labels == PERSON_CLASS_ID
            boxes = boxes[mask_person]
            scores = scores[mask_person]

            keep_mask = scores >= CONF_THRESH
            boxes = boxes[keep_mask]
            scores = scores[keep_mask]

            if len(boxes) > 0:
                keep_idxs = non_max_suppression(boxes, scores, iou_threshold=0.5)
                boxes = boxes[keep_idxs]
                scores = scores[keep_idxs]

                for bb, sc in zip(boxes, scores):
                    x1, y1, x2, y2 = bb
                    draw_transparent_box(frame, (x1, y1, x2, y2), "person", float(sc), box_color=(0, 200, 0),
                                         alpha=0.18)
                    w, h = x2 - x1, y2 - y1
                    coco_boxes.append({"image_id": frame_idx,
                                       "category_id": PERSON_CLASS_ID,
                                       "bbox": [float(x1), float(y1), float(w), float(h)],
                                       "score": float(sc)
                                       })
            avg_inf = total_time / processed if processed > 0 else 0.0
            info_text = f"Inf time/frame: {avg_inf:.3f}s, device: {DEVICE}"
            cv2.putText(frame, info_text, (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                        cv2.LINE_AA)
            out.write(frame)

    finally:
        cap.release()
        out.release()
    avg_time_per_frame = total_time / frame_idx if frame_idx > 0 else 0
    return coco_boxes, avg_time_per_frame