import os

def download():
    print("⬇ Downloading dataset from Kaggle...")

    os.makedirs("dataset", exist_ok=True)

    result = os.system(
        "kaggle datasets download -d iamsouravbanerjee/animal-image-dataset-90-different-animals -p dataset --unzip"
    )

    if result != 0:
        print("❌ Download failed! Check Kaggle API setup.")
    else:
        print("✅ Download successful!")

if __name__ == "__main__":
    download()