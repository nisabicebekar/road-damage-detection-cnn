from utils.rdd_dataset import RDDDataset

train_dataset = RDDDataset(
    image_dir="data/raw/RDD_SPLIT/train/images",
    label_dir="data/raw/RDD_SPLIT/train/labels"
)

print("Toplam train görüntü:", len(train_dataset))

img, label = train_dataset[0]
print("Image shape:", img.shape)
print("Label:", label)