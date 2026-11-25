import os
from huggingface_hub import HfApi
from dental_core.core.onnx_exporter import (
    export_yolov8_to_onnx,
    export_classifier_to_onnx,
)

MODEL_TYPE = "classifier"
REPO_ID = "chemahc94/Dental_013"
PT_PATH = "models/best_restoration_model.pth"
ONNX_OUT = "models/best_restoration_model.onnx"
NUM_CLASSES = 7


def main():
    if not os.path.exists(PT_PATH):
        print(f"[Dental_013] No .pth file found at {PT_PATH}.")
        return

    print(f"[Dental_013] Exporting {PT_PATH} (EfficientNet-B0 classifier) to ONNX...")

    onnx_path = None
    if MODEL_TYPE == "yolo":
        onnx_path = export_yolov8_to_onnx(PT_PATH)
    else:
        import timm
        import torch

        model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=NUM_CLASSES)
        state = torch.load(PT_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        dummy_input = torch.randn(1, 3, 224, 224)
        onnx_path = export_classifier_to_onnx(model, dummy_input, ONNX_OUT)

    if onnx_path and os.path.exists(onnx_path):
        print(f"[Dental_013] Uploading to HuggingFace {REPO_ID}...")
        api = HfApi()
        # Ensure you are logged in via `huggingface-cli login`
        try:
            api.upload_file(
                path_or_fileobj=onnx_path,
                path_in_repo="weights/best.onnx",
                repo_id=REPO_ID,
                repo_type="model",
            )
            print(
                f"[Dental_013] Upload successful. Deleting local weights to save space..."
            )
            os.remove(PT_PATH)
            os.remove(onnx_path)
            print(f"[Dental_013] Cleaned up local disk.")
        except Exception as e:
            print(f"[Dental_013] Upload failed: {e}")


if __name__ == "__main__":
    main()
