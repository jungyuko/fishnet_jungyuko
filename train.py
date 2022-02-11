import argparse
import logging

import numpy as np

import torch
import torch.nn as nn

from models.fishnet import build_fishnet
from utils import prepare_dataloader, config_logging, BestSaver, AverageMeter

from matplotlib import pyplot as plt

from tqdm import tqdm


device = torch.device("cuda:0")
print(device)


parser = argparse.ArgumentParser(description='FishNet')

parser.add_argument("--img_size", type=int, default=32, help="img_size")

# Model Parameters
parser.add_argument("--batch_size", type=int, default=256, help="number of batch")
parser.add_argument("--n_tail", type=int, default=3, help="number of tail modules")
parser.add_argument("--n_body", type=int, default=3, help="number of body modules")
parser.add_argument("--n_head", type=int, default=3, help="number of head modules")
parser.add_argument("--in_channel", type=int, default=64, help="dimension of channels")
parser.add_argument("--out_channel", type=int, default=64, help="dimension of channels")

# Training Parameters
parser.add_argument("--lr", type=float, default=1e-1, help="initial learning rate")
parser.add_argument("--wd", type=float, default=1e-4, help="initial learning rate")
parser.add_argument("--momentum", type=float, default=0.9, help="initial learning rate")
parser.add_argument("--epochs", type=int, default=50, help="number of max epochs")

# name
parser.add_argument("--comment", type=str, default='model', help="save pth name")

args = parser.parse_args('')
print(args)


comment = args.comment
## Logger
config_logging(comment)
logging.info("arguments: {}".format(args))


## Dataloader
train_dataset, train_loader, valid_dataset, valid_loader, _, _ = prepare_dataloader(img_size=args.img_size)

print('length of batch size, train data: {}'.format(len(train_loader)))
print('length of batch size, valid data: {}'.format(len(valid_loader)))

## model
model = build_fishnet(args)
model = model.to(device)


total_train_losses = []
total_train_acc = []
total_valid_acc = []

def train(model, device, train_loader, valid_loader):
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, 
                                weight_decay=args.wd, momentum=args.momentum)

    epochs = args.epochs
    saver = BestSaver(comment)

    for epoch in tqdm(range(1, epochs+1)):
        logging.info("Train Phase, Epoch: {}/{}".format(epoch, epochs))
        train_losses = AverageMeter()
        
        # Train Phase
        model.train()

        correct = 0
        count   = 0
        for batch_num, data in enumerate(train_loader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            pred_labels = np.argmax(output.cpu().detach().numpy(), axis=1)

            # loss
            train_loss = criterion(output, labels)
            train_losses.update(train_loss.item(), images.shape[0])


            for i in range(len(labels)):
                if labels[i] == pred_labels[i]:
                    correct += 1
                    count   += 1
                else:
                    count += 1

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            if batch_num % 20 == 0:
                logging.info(
                    "[{}/{}] # {}/{} loss: {:.4f}".format(epoch, epochs, batch_num, len(train_loader), train_losses.val)
                )
        
        logging.info("Train_loss: {:.4f}, Train_acc: {:.4f}%".format(train_losses.avg, (correct/count)*100))
        total_train_losses.append(train_losses.avg)
        total_train_acc.append(correct/count)

        # Valid Phase
        model.eval()

        correct = 0
        count   = 0
        for batch_num, data in enumerate(valid_loader):
            images, gt_labels = data

            images    = images.to(device)
            gt_labels = gt_labels.to(device)
            
            with torch.no_grad():
                output = model(images)
            pred_labels = np.argmax(output.cpu().detach().numpy(), axis=1)
            
            for i in range(len(gt_labels)):
                if gt_labels[i] == pred_labels[i]:
                    correct += 1
                    count   += 1
                else:
                    count += 1

        logging.info('valid_acc: {:.4f}%'.format((correct/count)*100))
        total_valid_acc.append(correct/count)
        saver.save(correct/count, model.state_dict())    

if __name__ == "__main__":
    train(model, device, train_loader, valid_loader)   