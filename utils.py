import os
import logging

import torch
import torchvision
import numpy as np
from dataloader.dataloader import CIFAR10


## color augmentation
## https://github.com/kevin-ssy/FishNet/blob/master/utils/data_aug.py #4
class ColorAugmentation(object):
    def __init__(self):
        self.eig_vec = torch.Tensor([
            [0.4009, 0.7192, -0.5675],
            [-0.8140, -0.0045, -0.5808],
            [0.4203, -0.6948, -0.5836],
        ])
        self.eig_val = torch.Tensor([[0.2175, 0.0188, 0.0045]])

    def __call__(self, tensor):
        assert tensor.size(0) == 3
        alpha = torch.normal(mean=torch.zeros_like(self.eig_val)) * 0.1
        quatity = torch.mm(self.eig_val * alpha, self.eig_vec)
        tensor = tensor + quatity.view(3, 1, 1)
        return tensor


class BestSaver(object):

    def __init__(self, comment=None):
        exe_fname=os.path.basename(__file__)
        save_path = "{}".format(exe_fname.split(".")[0])
        
        if comment is not None and str(comment):
            save_path = save_path + "_" + str(comment)

        save_path = save_path + ".pth"

        self.save_path = save_path
        self.best = float('-inf')

    def save(self, metric, data):
        if metric >= self.best:
            self.best = metric
            torch.save(data, self.save_path)
            logging.info("Saved best model to {}".format(self.save_path))

class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val, self.avg, self.sum, self.count = 0.0, 0.0, 0.0, 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def config_logging(comment=None):

    if comment is not None and str(comment):
        log_fname = str(comment)

    log_fname = log_fname + ".log"
    log_format = "%(asctime)s [%(levelname)-5.5s] %(message)s"

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[logging.FileHandler(log_fname), logging.StreamHandler()]
    )


def prepare_dataloader(
            img_size=32):

    img_size = img_size
    
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(img_size),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        ColorAugmentation(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    ])
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    ])
       
    train_dataset = CIFAR10(img_path='./data/train/',
                        json_path='./data/',
                        json_file='train_img2labels.json',
                        transform=train_transform)
    train_loader = torch.utils.data.DataLoader(
                            train_dataset, 
                            batch_size=256, 
                            shuffle=True
    )

    valid_dataset = CIFAR10(img_path='./data/valid/',
                        json_path='./data/',
                        json_file='valid_img2labels.json',
                        transform=test_transform)
    valid_loader = torch.utils.data.DataLoader(
                            valid_dataset, 
                            batch_size=256, 
                            shuffle=False
    )

    test_dataset = CIFAR10(img_path='./data/test/',
                        json_path='./data/',
                        json_file='test_img2labels.json',
                        transform=test_transform)
    test_loader = torch.utils.data.DataLoader(
                            test_dataset, 
                            batch_size=1, 
                            shuffle=False
    )

    return train_dataset, train_loader, valid_dataset, valid_loader, test_dataset, test_loader
