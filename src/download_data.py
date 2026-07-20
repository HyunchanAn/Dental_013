import os
import subprocess
import zipfile


def download_dataset():
    dataset_name = "lokisilvres/dental-disease-panoramic-detection-dataset"
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, "data", "raw")

    os.makedirs(raw_dir, exist_ok=True)

    print(f"Downloading {dataset_name} from Kaggle...")
    try:
        # Require kaggle CLI to be installed
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset_name, "-p", raw_dir],
            check=True,
        )
        print("Download complete.")

        # Unzip
        zip_path = os.path.join(
            raw_dir, "dental-disease-panoramic-detection-dataset.zip"
        )
        if os.path.exists(zip_path):
            print("Extracting zip file...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(raw_dir)
            print("Extraction complete.")
            os.remove(zip_path)  # Cleanup
        else:
            print(f"Zip file not found at {zip_path}")

    except subprocess.CalledProcessError:
        print("Error: Failed to download dataset via Kaggle API.")
        print(
            "Please ensure you have installed the kaggle library (`pip install kaggle`)"
        )
        print("and have your kaggle.json token placed in ~/.kaggle/kaggle.json")
    except FileNotFoundError:
        print("Error: 'kaggle' command not found. Please run: pip install kaggle")


if __name__ == "__main__":
    download_dataset()
