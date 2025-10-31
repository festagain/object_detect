import cv2
import torch
import numpy as np
import torchvision
from config import PERSON_CLASS_ID, CONF_THRESH

def non_max_suppression(boxes, scores, iou_threshold=0.5):
    """
       Применяет Non-Maximum Suppression (NMS) к списку bounding box'ов.

       Args:
           boxes (list or np.ndarray or torch.Tensor): координаты bbox [x1, y1, x2, y2], shape [N, 4].
           scores (list or np.ndarray or torch.Tensor): оценки (confidences) для каждого bbox, shape [N].
           iou_threshold (float, optional): порог IoU для удаления перекрывающихся bbox.
       Returns:
           np.ndarray: индексы bbox, которые остались после NMS.
       """

    if len(boxes) == 0:
        return []
    idxs = torchvision.ops.nms(
        torch.tensor(boxes).float(),
        torch.tensor(scores).float(),
        iou_threshold
    )
    return idxs.numpy()

def draw_transparent_box(img, xyxy, label_text, score, box_color=(0, 255, 0), alpha=0.25):
    """
    Рисует полупрозрачный bounding box с подписью на изображении.

    Args:
        img (np.ndarray): изображение BGR.
        xyxy (tuple/list): координаты bbox (x1, y1, x2, y2).
        label_text (str): подпись для объекта (например, "person").
        score (float): confidence score объекта.
        box_color (tuple, optional): цвет bbox в формате (B, G, R). Default: зелёный.
        alpha (float, optional): прозрачность заливки bbox. Default: 0.25.

    Returns:
        None. Рисует на переданном изображении in-place.
    """
    x1, y1, x2, y2 = map(int, xyxy)
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)

    text = f"{label_text}: {score:.2f}"
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    tx, ty = x1, max(0, y1 - 6)
    cv2.rectangle(img, (tx, ty - th - baseline), (tx + tw, ty + baseline), (0, 0, 0), -1)
    cv2.putText(img, text, (tx, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)