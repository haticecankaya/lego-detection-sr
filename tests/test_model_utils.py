import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from PIL import Image
from src.model_utils import load_sr_model, load_detection_model, enhance_image

# 1. Test: YOLO Modeli yükleme testi (Mock ile)
@patch('src.model_utils.YOLO')
def test_load_detection_model(mock_yolo):
    mock_yolo.return_value = "FakeModel"
    model = load_detection_model()
    assert model == "FakeModel"
    mock_yolo.assert_called_once()

# 2. Test: YOLO Modeli bulunamazsa None dönmeli
@patch('src.model_utils.YOLO')
def test_load_detection_model_fail(mock_yolo):
    mock_yolo.side_effect = Exception("Model not found")
    model = load_detection_model()
    assert model is None

# 3. Test: SR Modeli yükleme testi (Mock ile)
@patch('src.model_utils.RRDBNet')
@patch('torch.load')
@patch('src.model_utils.RealESRGANer')
def test_load_sr_model(mock_realesrganer, mock_torch_load, mock_rrdbnet):
    mock_torch_load.return_value = {'params_ema': {}}
    mock_realesrganer.return_value = "FakeUpsampler"
    
    model = load_sr_model()
    assert model == "FakeUpsampler"

# 4. Test: Görüntü netleştirme fonksiyonu çalışıyor mu?
def test_enhance_image_success():
    
    mock_upsampler = MagicMock()
   
    fake_output = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_upsampler.enhance.return_value = (fake_output, None)
    
    input_img = Image.new('RGB', (25, 25))
    result = enhance_image(mock_upsampler, input_img)
    
    assert isinstance(result, Image.Image)
    assert result.size == (100, 100)

# 5. Test: Netleştirme sırasında hata olursa orijinal resim dönmeli
def test_enhance_image_failure():
    mock_upsampler = MagicMock()
    mock_upsampler.enhance.side_effect = Exception("Processing Error")
    
    input_img = Image.new('RGB', (25, 25))
    result = enhance_image(mock_upsampler, input_img)
    
    assert result == input_img