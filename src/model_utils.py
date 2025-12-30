import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from ultralytics import YOLO
import numpy as np
from PIL import Image
from src.config import SR_MODEL_PATH, YOLO_MODEL_PATH
from src.utils import get_logger

logger = get_logger(__name__)

def load_sr_model():
    """RealESRGAN modelini yükler."""
    try:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        state_dict = torch.load(SR_MODEL_PATH, map_location=torch.device('cpu'))['params_ema']
        model.load_state_dict(state_dict, strict=True)
        upsampler = RealESRGANer(scale=4, model_path=SR_MODEL_PATH, model=model, tile=0, pre_pad=0, half=False)
        logger.info("Süper çözünürlük modeli başarıyla yüklendi.")
        return upsampler
    except FileNotFoundError:
        logger.error(f"Model dosyası bulunamadı: {SR_MODEL_PATH}")
        return None
    except Exception as e:
        logger.error(f"SR Model yükleme hatası: {e}")
        return None

def load_detection_model():
    """YOLO modelini yükler."""
    try:
        model = YOLO(YOLO_MODEL_PATH)
        logger.info("YOLO modeli başarıyla yüklendi.")
        return model
    except Exception as e:
        logger.error(f"YOLO Model yükleme hatası: {e}")
        return None

def enhance_image(upsampler, crop_img):
    """Görüntüyü netleştirir."""
    try:
        img_array = np.array(crop_img)
        output, _ = upsampler.enhance(img_array, outscale=4)
        return Image.fromarray(output)
    except Exception as e:
        logger.error(f"Görüntü işleme hatası: {e}")
        return crop_img  