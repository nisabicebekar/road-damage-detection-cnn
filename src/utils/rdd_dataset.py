import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class RDDDataset(Dataset):
    def __init__(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.images = os.listdir(image_dir)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)

        label_name = img_name.replace(".jpg", ".txt")
        label_path = os.path.join(self.label_dir, label_name)

        image = Image.open(img_path).convert("RGB")

        # YOLO label mantığı → binary
        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            label = 1  # damage
        else:
            label = 0  # no_damage

        return self.transform(image), label