import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_model():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "processed")

    if not os.path.exists(os.path.join(data_dir, "train")):
        print(
            f"Error: Training data not found in {data_dir}. Please run prepare_data.py first."
        )
        return

    # Data augmentation and normalization for training
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ["train", "val"]
    }

    # RTX 5080 16GB VRAM + Ryzen 9900X 최적화
    dataloaders = {
        x: DataLoader(
            image_datasets[x],
            batch_size=128,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )
        for x in ["train", "val"]
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
    class_names = image_datasets["train"].classes
    print(f"Classes: {class_names}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Use EfficientNet-B0
    model_ft = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    num_ftrs = model_ft.classifier[1].in_features
    model_ft.classifier[1] = nn.Linear(num_ftrs, len(class_names))
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_ft.parameters(), lr=0.001)

    # Extended training loop with Early Stopping
    num_epochs = 100
    best_loss = float("inf")
    best_acc = 0.0
    patience = 15
    epochs_no_improve = 0

    os.makedirs(os.path.join(base_dir, "models"), exist_ok=True)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model_ft.train()
            else:
                model_ft.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase], desc=phase):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model_ft(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # Early Stopping and Save best model based on val loss
            if phase == "val":
                if epoch_acc > best_acc:
                    best_acc = epoch_acc

                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    epochs_no_improve = 0
                    torch.save(
                        model_ft.state_dict(),
                        os.path.join(base_dir, "models", "best_restoration_model.pth"),
                    )
                else:
                    epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    print(
        f"Training complete. Best Val Loss: {best_loss:.4f}, Best Val Acc: {best_acc:.4f}"
    )


if __name__ == "__main__":
    train_model()
