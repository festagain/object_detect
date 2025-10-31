import torch
import torchvision
from torch.optim import AdamW

from config import DEVICE, LR

def get_model(num_classes=2, pretrained=True):
    """Загружает предобученную Faster R-CNN модель.
    Args:
        num_classes (int): Количество классов
        pretrained (bool): Флаг, который показывает нужны ли предобученные веса
    Returns:
        model (torch.nn.Module): Модель Faster R-CNN, переведённая на DEVICE (CPU/GPU), готовая к обучению.
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights="DEFAULT" if pretrained else None
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )
    # Заморозка backbone
    for param in model.backbone.parameters():
        param.requires_grad = False
    return model.to(DEVICE)

def train_model(model, train_loader, num_epochs=5):
    """
    Обучает Faster R-CNN модель на даталоадере train_loader.

    Args:
        model (torch.nn.Module): Модель, подготовленная через get_model().
        train_loader (torch.utils.data.DataLoader): Даталоадер с батчами (images, targets).
        num_epochs (int, optional): Количество эпох обучения.

    Returns:
        None. Выводит среднюю потерю по каждой эпохе.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=LR, weight_decay=1e-4)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for imgs, targets in train_loader:
            imgs = [img.to(DEVICE) for img in imgs]
            targets = [{k: v.to(DEVICE) for k, v in t.items() if k in ["boxes", "labels"]}
                       for t in targets]

            loss_dict = model(imgs, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            total_loss += losses.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")