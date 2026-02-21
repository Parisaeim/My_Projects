import ssl
import certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
import torch
import os
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torchvision.models as models
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
from datasets import load_dataset
from sklearn.metrics.pairwise import euclidean_distances
from torchvision import datasets, transforms, models
#image_folder = 'C:/Users/Sharif/PycharmProjects/PythonProject1/PetImages'
#images = os.listdir(image_folder)

resnet50 = models.resnet50(pretrained=True)
resnet50 = nn.Sequential(*list(resnet50.children())[:-1])
resnet50.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet50 = resnet50.to(device)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])
])

test_image= 'test.jpg'
test_image = Image.open(test_image).convert('RGB')
test_image = transform(test_image).unsqueeze(0).to(device)

with torch.no_grad():
    test_features = resnet50(test_image)
    test_features = test_features.view(test_features.size(0), -1)
    test_features = test_features.cpu().numpy()
dir_features = np.load('features.npy')
image_path = np.load('Image_path.npy')

distances = euclidean_distances(test_features, dir_features)
nearest_image_index = np.argmin(distances)
nearest_image_path = image_path[nearest_image_index]

print (f'The nearest image from dataset to test image is :{nearest_image_path}')
