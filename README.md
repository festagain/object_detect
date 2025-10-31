# Детекция объектов на видео с использованием Faster R-CNN и PyTorch

## О проекте
Проект реализует детекцию объектов (например, людей) на видео с использованием **Faster R-CNN** и **PyTorch**.

## Структура проекта
object_detect/
├── main.py # Точка входа: обучение и инференс модели
├── dataset.py # Класс VideoFramesDataset
├── split_images.py # Скрипт для разбиения видео на кадры
├── metric.py # Скрипт для вычисления метрик COCO
├── utils.py # Вспомогательные функции (NMS, отрисовка bbox)
├── config.py # Конфигурация: пути, устройство, гиперпараметры
├── requirements.txt
└── README.md

## Скрипты
- `split_images.py` – разбивает видео на датасет кадров.  
- `metric.py` – вычисляет метрики через:
  - `instances_scaled.json` — аннотации (отскейленное видео)  
  - `results.json` — предсказания модели

## Запуск
- Для обучения модели:
```bash
python main.py --mode train
```
- Для инференса модели:
```bash
python main.py --mode infer
```
