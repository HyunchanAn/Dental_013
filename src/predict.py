import os
from typing import Dict, List, Optional, Sequence, Union
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

DEFAULT_CROP_CLASS_NAMES: List[str] = [
    "Bridge",
    "Crown",
    "Filling",
    "Implant",
    "Natural",
    "Other",
    "RCT",
]

CROP_CLASS_TO_SSOT_TAG: Dict[str, Optional[str]] = {
    "Bridge": "Crown/Bridge",
    "Crown": "Crown/Bridge",
    "Filling": "Inlay/Onlay",
    "Implant": "Implant",
    "RCT": "Endodontic (RCT)",
    "Natural": None,
    "Other": None,
}

_CROP_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def map_raw_restoration_to_ssot(raw_label: str) -> Optional[str]:
    if raw_label in CROP_CLASS_TO_SSOT_TAG:
        return CROP_CLASS_TO_SSOT_TAG[raw_label]
    low = (raw_label or "").strip().lower()
    if low in ("filling", "inlay", "onlay", "inlay/onlay", "amalgam", "resin"):
        return "Inlay/Onlay"
    if low in ("crown", "bridge", "crown/bridge", "metal band", "plating"):
        return "Crown/Bridge"
    if low in ("implant", "tad", "abutment", "gingival former"):
        return "Implant"
    if low in ("rct", "root canal treatment", "post - core", "endodontic (rct)"):
        return "Endodontic (RCT)"
    return None


def load_model(model_path: str, num_classes: int = 7):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    if os.path.exists(model_path):
        try:
            state = torch.load(model_path, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model = model.to(device)
        model.eval()
        return model, device
    print(f"Model not found at {model_path}")
    return None, device


def predict_crop_array(
    model: nn.Module,
    device: torch.device,
    crop_image: Union[np.ndarray, Image.Image],
    class_names: Sequence[str] = DEFAULT_CROP_CLASS_NAMES,
    is_bgr: bool = True,
) -> Dict[str, Union[str, float, Optional[str], Dict[str, float]]]:
    """Run EfficientNet-B0 inference on an in-memory tooth crop array or PIL image."""
    if isinstance(crop_image, Image.Image):
        pil_img = crop_image.convert("RGB")
    else:
        arr = np.asarray(crop_image)
        if arr.ndim == 2:
            rgb = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 3 and is_bgr:
            rgb = arr[:, :, ::-1]
        else:
            rgb = arr[:, :, :3]
        pil_img = Image.fromarray(rgb.astype(np.uint8)).convert("RGB")

    img_tensor = _CROP_TRANSFORM(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        max_prob, pred_idx_t = torch.max(probs, 0)

    class_idx = int(pred_idx_t.item())
    raw_cls = class_names[class_idx] if class_idx < len(class_names) else "Unknown"
    ssot_tag = map_raw_restoration_to_ssot(raw_cls)
    prob_dict = {
        class_names[i]: round(float(probs[i].item()), 4)
        for i in range(min(len(class_names), len(probs)))
    }
    return {
        "class": raw_cls,
        "clinical_category": ssot_tag,
        "confidence": float(max_prob.item()),
        "probabilities": prob_dict,
    }


def predict_crop(model, device, image_path: str, class_names: Sequence[str] = DEFAULT_CROP_CLASS_NAMES):
    image = Image.open(image_path).convert("RGB")
    res = predict_crop_array(model, device, image, class_names=class_names, is_bgr=False)
    return {
        "class": res["class"],
        "clinical_category": res["clinical_category"],
        "confidence": res["confidence"],
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_cropped_image>")
        sys.exit(1)

    img_path = sys.argv[1]
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "best_restoration_model.pth")

    model, device = load_model(model_path, len(DEFAULT_CROP_CLASS_NAMES))
    if model:
        result = predict_crop(model, device, img_path, DEFAULT_CROP_CLASS_NAMES)
        print(f"Prediction: {result['class']} -> {result['clinical_category']} ({result['confidence']:.4f})")
