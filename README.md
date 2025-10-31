Проект реализует детекцию объектов (например, людей) на видео с использованием Faster R-CNN и PyTorch.
Структура проекта:
    object_detect/
    ├── main.py            # Точка входа: обучение и инференс модели
    ├── dataset.py         # Класс VideoFramesDataset
    ├── split_images.py    # Скрипт для разбиения видео на кадры
    ├── metric.py          # Скрипт для вычисления метрик COCO
    ├── utils.py           # Вспомогательные функции (NMS, отрисовка bbox)
    ├── config.py          # Конфигурация: пути, устройство, гиперпараметры
    ├── requirements.txt
    └── README.md
split_images.py разбивает видео на датасет фотографий по фреймам
metric.py вычисляет метрику через instances_scaled.json - отскейлинная версия аннотация и results.json(выход модели)
запуск для обучения - python main.py --mode train
запуск для инференса - python main.py --mode infer
