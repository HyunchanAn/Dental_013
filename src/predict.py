import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


def load_model(model_path, num_classes):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        return model, device
    else:
        print(f"Model not found at {model_path}")
        return None, device


def predict_crop(model, device, image_path, class_names):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        max_prob, preds = torch.max(probabilities, 0)

    class_idx = preds.item()
    return {
        "class": class_names[class_idx] if class_idx < len(class_names) else "Unknown",
        "confidence": max_prob.item(),
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_cropped_image>")
        sys.exit(1)

    img_path = sys.argv[1]

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "best_restoration_model.pth")

    # Needs to match the classes folder names exactly in order they were trained
    # This should ideally be loaded from a saved mapping or classes.json
    class_names = ["Bridge", "Crown", "Filling", "Implant", "Natural", "Other", "RCT"]

    model, device = load_model(model_path, len(class_names))
    if model:
        result = predict_crop(model, device, img_path, class_names)
        print(f"Prediction: {result['class']} (Confidence: {result['confidence']:.4f})")
