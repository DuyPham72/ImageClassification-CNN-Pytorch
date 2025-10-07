from dataset import MyAnimalDataset
from model import MyCNN
import os
import torch
import shutil
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, Resize, ToTensor, RandomResizedCrop, RandomHorizontalFlip, RandomRotation, ColorJitter
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm.autonotebook import tqdm
import matplotlib

matplotlib.use("Agg")

def plot_confusion_matrix(writer, cm, class_names, epoch):
    figure = plt.figure(figsize=(16, 16))
    # color map: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    plt.imshow(cm, interpolation='nearest', cmap="YlOrBr")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)
    plt.close(figure)

def args_parser():
    parser = argparse.ArgumentParser(prog='Training CNN Model Input Parser')

    parser.add_argument('-p', '--path', type=str, default='./animals')
    parser.add_argument('-b', '--batch_size', type=int, default=16)
    parser.add_argument('-l', '--lr', type=float, default=1e-3)
    parser.add_argument('-m', '--momentum', type=float, default=0.9)
    parser.add_argument('-e', '--epoch', type=int, default=100)
    parser.add_argument('-t', '--tensorboard', type=str, default='my_tensorboard')
    parser.add_argument('-r', '--resume', type=bool, default=False)
    
    args = parser.parse_args()
    return args

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up transformer for train and validation
    train_transform = Compose([
        RandomResizedCrop(224),
        RandomHorizontalFlip(),
        RandomRotation(10),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ToTensor()
    ])
    val_transform = Compose([
        Resize(size=(224, 224)),
        ToTensor()
    ])

    # Load datasets
    train_dataset = MyAnimalDataset(path_dir=args.path, is_train=True, transform=train_transform)
    val_dataset = MyAnimalDataset(path_dir=args.path, is_train=False, transform=val_transform)

    # Load datasets into dataloaders
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True
    )

    # Declare tensorboard writer
    if os.path.isdir(args.tensorboard):
        shutil.rmtree(path=args.tensorboard)
    os.mkdir(path=args.tensorboard)
    writer = SummaryWriter(log_dir=args.tensorboard)

    # Declare model, criterion, optimizer, schedular for training
    model = MyCNN(len(train_dataset.catagory)).to(device=device, non_blocking=True)
    criterion = CrossEntropyLoss()
    optimizer = AdamW(params=model.parameters(), lr=args.lr)        
    scheduler = ReduceLROnPlateau(optimizer, mode='min')            
    max_iteration = len(train_dataloader)

    # Load model, optimizer weight
    if args.resume:
        checkpoint = torch.load(f='curr_model.pt')
        model.load_state_dict(state_dict=checkpoint['model'])
        optimizer.load_state_dict(state_dict=checkpoint['optimizer'])
        scheduler.load_state_dict(state_dict=checkpoint['scheduler'])
        start_epoch = checkpoint['start_epoch']
        best_accuracy = checkpoint['best_accuracy']
    else:
        start_epoch = 0
        best_accuracy= -1

    # Start Training
    for epoch in range(start_epoch, args.epoch):
        # Train Mode
        model.train()
        loss_stored = []

        train_progession_bar = tqdm(train_dataloader)
        for idx, (images, labels) in enumerate(train_progession_bar):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            prediction = model(images)
            loss = criterion(prediction, labels)
            loss_stored.append(loss.item())
            avg_loss = np.mean(loss_stored)

            train_progession_bar.set_description(desc=f'Loss: {avg_loss:.4f}')
            writer.add_scalar(tag='Train/Loss', scalar_value=avg_loss, global_step=(epoch*max_iteration)+idx)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation Mode
        model.eval()
        loss_ls = []
        prediction_ls = []
        labels_ls = []

        val_progession_bar = tqdm(val_dataloader)
        with torch.inference_mode():
            for images, labels in val_progession_bar:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                prediction = model(images)
                loss = criterion(prediction, labels)
                loss_ls.append(loss.item())

                predicted_class = torch.argmax(prediction, dim=1)
                prediction_ls.extend(predicted_class.cpu())
                labels_ls.extend(labels.cpu())

        avg_loss = np.mean(loss_ls)
        acc = accuracy_score(labels_ls, prediction_ls)
        print(f"Validation. Epoch: {epoch+1}/{args.epoch}. Average loss: {avg_loss:.4f}. Accuracy: {acc:.4f}")

        # Update lr of schedular
        scheduler.step(avg_loss)        

        writer.add_scalar(tag='Val/Loss', scalar_value=avg_loss, global_step=epoch)
        writer.add_scalar(tag='Val/Accuracy', scalar_value=acc, global_step=epoch)
        plot_confusion_matrix(writer=writer, cm=confusion_matrix(labels_ls, prediction_ls), class_names=val_dataset.catagory, epoch=epoch)

        # Save checkpoint for further training
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'start_epoch': epoch+1,
            'best_accuracy': best_accuracy,
        }
        torch.save(obj=checkpoint, f='curr_model.pt')

        # Save checkpoint of best model to deploy
        if acc > best_accuracy:
            torch.save(obj=model.state_dict(), f='best_model.pt')
            best_accuracy = acc

    writer.close()

if __name__ == '__main__':
    args = args_parser()
    train(args=args)