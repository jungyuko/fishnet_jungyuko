import os
import json
import numpy as np

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset

from PIL import Image
from matplotlib import pyplot as plt


## DataLoader ##
## https://github.com/pytorch/vision/blob/main/torchvision/datasets/cifar.py ##
class CIFAR10(Dataset):
    def __init__(self, 
                img_path='../data/train/', 
                json_path='../data/',
                json_file='train_img2labels.json',
                transform=None):
        
        self.img_path  = img_path
        self.json_path = json_path
        self.json_file = json_file
        self.transform = transform
        
        self.data = json.load(open(os.path.join(self.json_path+self.json_file)))
        self.data = [(k,v) for k, v in self.data.items()]

    def __getitem__(self, index):
        
        img_name, label = self.data[index]
        
        img = self.img_path + img_name
        img = Image.open(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, label
        
    def __len__(self):
        return len(self.data)
    

def collate_fn(data):
    """Need custom a collate_fn"""
    data.sort(key=lambda x:x[0].shape[0], reverse=True)
    images, labels = zip(*data)

    images = torch.stack(images)
    labels = torch.LongTensor(labels)

    return images, labels


## visualize dataset
def imshow(img):
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1,2,0)))
    plt.axis('off')
    plt.show()


## execute dataloader  
if __name__ == "__main__":
    
    img_size = 32
    batch_size = 8
    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(img_size),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
    ])

    idx2class = {
            0: 'airplane',
            1: 'automobile',
            2: 'bird',
            3: 'cat', 
            4: 'deer',
            5: 'dog',
            6: 'frog',
            7: 'horse',
            8: 'ship',
            9: 'truck'
            }
        
    d = CIFAR10(img_path='../data/train/',
                json_path='../data/',
                transform=transform)
    loader = DataLoader(d, batch_size, shuffle=False, collate_fn=collate_fn)
    
    ## visualize cifar10 dataset
    for i, data in (enumerate(loader)):
        img, label = data
        lst_label = label.tolist()
        gt_label = []
        for j in range(len(lst_label)):
            gt_label.append(idx2class[lst_label[j]])
        
        imshow(torchvision.utils.make_grid(img))
        print(gt_label)
        if i == 10:
            break