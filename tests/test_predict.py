import os
import sys
import torch
from PIL import Image

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import predict


def test_load_model_missing_weight_returns_none(tmp_path):
    missing_path = str(tmp_path / "nonexistent_model.pth")
    model, device = predict.load_model(missing_path, num_classes=7)
    assert model is None
    assert isinstance(device, torch.device)


def test_load_model_and_predict_crop_interface(tmp_path):
    class_names = ["Bridge", "Crown", "Filling", "Implant", "Natural", "Other", "RCT"]
    dummy_weights_path = str(tmp_path / "dummy_restoration.pth")

    # Construct an uninitialized EfficientNet-B0 with 7 classes and save its state_dict
    from torchvision import models
    import torch.nn as nn

    base_model = models.efficientnet_b0(weights=None)
    num_ftrs = base_model.classifier[1].in_features
    base_model.classifier[1] = nn.Linear(num_ftrs, len(class_names))
    torch.save(base_model.state_dict(), dummy_weights_path)

    loaded_model, device = predict.load_model(dummy_weights_path, len(class_names))
    assert loaded_model is not None

    dummy_img_path = str(tmp_path / "crop.jpg")
    Image.new("RGB", (128, 128), color=(120, 120, 120)).save(dummy_img_path)

    result = predict.predict_crop(loaded_model, device, dummy_img_path, class_names)
    assert isinstance(result, dict)
    assert result["class"] in class_names
    assert 0.0 <= result["confidence"] <= 1.0
