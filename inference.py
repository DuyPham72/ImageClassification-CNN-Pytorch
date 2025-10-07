from model import MyCNN
import torch
import argparse
import numpy as np
from PIL import Image

def deploy(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Declare all available classes
    categories = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider", "squirrel"]

    # Declare model and load weights
    model = MyCNN(len(categories)).to(device)
    checkpoint = torch.load(f='best_cnn_model.pt')
    model.load_state_dict(checkpoint)
    
    # Open sample image and convert it to tensor
    image = Image.open(args.image).convert('RGB')
    image.resize(size=(224, 224))
    image = np.transpose(image, (2, 0, 1))/255
    image = torch.from_numpy(image)[None, :, :, :].float().to(device)

    # Start testing
    model.eval()
    with torch.inference_mode():
        output = model(image)
        predict_input = torch.argmax(output, dim=1)

    # Print result of the model
    print(f"The image is about: {categories[predict_input]}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Inference CNN Model Input Parser')
    parser.add_argument('-i', '--image', type=str, default='./animals/test/cat/10.jpeg')
    args = parser.parse_args()

    deploy(args=args)