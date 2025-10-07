from dataset import MyAnimalDataset
import torch
import torch.nn as nn
from torchvision.transforms.v2 import Compose, Resize, ToImage, ToDtype
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

class MyCNN(nn.Module):
    def __init__(self, num_class):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv1 = self.create_conv_block(in_channel=64, out_channel=64, stride=1)
        self.conv2 = self.create_conv_block(in_channel=64, out_channel=128, stride=2)
        self.conv3 = self.create_conv_block(in_channel=128, out_channel=256, stride=2)
        self.conv4 = self.create_conv_block(in_channel=256, out_channel=512, stride=2)

        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = self.fc1 = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(512, num_class)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.gap(x)                 # Adaptive Avg Pooling
        x = x.view(x.shape[0], 512)     # same with last conv

        x = self.fc(x)
        return x
    
    def create_conv_block(self, in_channel, out_channel, kernel_size=3, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=1),
            nn.BatchNorm2d(num_features=out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channel),
            nn.ReLU(inplace=True)
        )
    
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = Compose([
        Resize(size=(224, 224)),
        ToImage(),
        ToDtype(dtype=torch.float32, scale=True)
    ])

    dataset = MyAnimalDataset('./animals', is_train=True, transform=transform)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )

    model = MyCNN(len(dataset.catagory)).to(device=device, non_blocking=True)
    progession_bar = tqdm(dataloader)
    for images, labels, in progession_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        prediction = model(images)
        progession_bar.set_description(f'shape: {prediction.shape}')