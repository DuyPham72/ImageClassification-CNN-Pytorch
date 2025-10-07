import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.v2 import Resize, ToDtype, ToImage, Compose
from PIL import Image

class MyAnimalDataset(Dataset):
    def __init__(self, path_dir, is_train=True, transform=None):
        path = os.path.join(path_dir, 'train' if is_train else 'test')

        self.catagory = sorted(os.listdir(path))
        self.images = []
        self.labels = []
        self.transform = transform

        for idx, dir_name in enumerate(self.catagory):
            dir_path = os.path.join(path, dir_name)
            for image in os.listdir(dir_path):
                image_path = os.path.join(dir_path, image)
                self.images.append(image_path)
                self.labels.append(idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        image = Image.open(image).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = Compose([
        Resize(size=(224,224)),
        ToImage(),
        ToDtype(dtype=torch.float32, scale=True)
    ])

    dataset = MyAnimalDataset(path_dir='./animals', is_train=True, transform=transform)
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        print(images.shape, labels.shape)