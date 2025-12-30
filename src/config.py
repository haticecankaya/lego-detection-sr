import os

# Dosya Yolları
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Model Dosya İsimleri
YOLO_MODEL_PATH = os.path.join(MODELS_DIR, "best.pt")
SR_MODEL_PATH = os.path.join(MODELS_DIR, "RealEsrgan_x4plus.pth")

# Uygulama Ayarları
PAGE_TITLE = "Lego Tespiti ve Görüntü Netleştirme"
LAYOUT = "wide"
IMAGE_WIDTH = 400