from pathlib import Path
import torch
#Пути к файлам
INPUT_VIDEO = str(Path("crowd.mp4").resolve())
OUTPUT_VIDEO = str(Path("crowd_final.mp4").resolve())
INPUT_FRAMES_DIR = "frames"
ANNOTATION_FILE = "instances_Validation.json"
#Настройки модели
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONF_THRESH = 0.5
PERSON_CLASS_ID = 1
SCALE = 2.0
DOWNSAMPLE_FOR_SPEED = 1
#Настройки обучения
BATCH_SIZE = 2
NUM_EPOCHS = 5
LR = 1e-4
MODEL_PATH = str(Path("model.pth").resolve())