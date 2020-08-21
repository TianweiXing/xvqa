"""
Author: Tianwei Xing (twxing@ucla.edu)
"""

import nltk
import pickle
import json

import torch
from torchvision import transforms
from src.transforms import Scale
from PIL import Image
from torchvision.models.resnet import ResNet, resnet101


def visualize_data(full_data, show = False):
    image_path = 'data/clevr/CLEVR_v1.0/images/' + full_data['split'] +'/' + full_data['image_filename']
    
    if show:
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        img=mpimg.imread(image_path)
        imgplot = plt.imshow(img)
        plt.show()
    
    print('Question:\t', full_data['question'])
    print('Answer:\t\t', full_data['answer'])
    
    return image_path, full_data['question'], full_data['answer']


def feature_processing(img_path, qst, img_data = None):
    
    # question:

    dataset_type = 'CLEVR'
    with open(f'data/{dataset_type}_dic.pkl', 'rb') as f:
        dic = pickle.load(f)
        
    result = []
    word_index = 1
    answer_index = 0
    
    words = nltk.word_tokenize(qst)
    question_token = []

    for word in words:
        question_token.append(dic['word_dic'][word])
        
    question = torch.tensor([question_token])
    q_len = len(question_token)

    
    # image
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x

    transform = transforms.Compose([
        Scale([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])])
    
       
    if img_data:
        img = img_data
    else:
        img = Image.open(img_path).convert('RGB')
    t_img = transform(img)
    t_img.unsqueeze_(0)
    if img_data:
        img = img_data
    
    t_img= t_img.to(device)

    resnet = resnet101(True).to(device)
    resnet.eval()
    resnet.forward = forward.__get__(resnet, ResNet)

    features = resnet(t_img).detach().cpu()#.numpy()

    
    return features, question, q_len
    