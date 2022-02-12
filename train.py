import argparse
import logging

import torch
import torch.nn as nn

from torch.optim import lr_scheduler

from models.fishnet import build_fishnet
from utils import prepare_dataloader, config_logging, BestSaver, AverageMeter


from tqdm import tqdm


## Use GPU
device = torch.device("cuda:0")
print(device)


## parameter setting
parser = argparse.ArgumentParser(description='FishNet')

parser.add_argument("--img_size", type=int, default=32, help="img_size")

# Model Parameters
parser.add_argument("--n_tail", type=int, default=3, help="number of tail modules")
parser.add_argument("--n_body", type=int, default=3, help="number of body modules")
parser.add_argument("--n_head", type=int, default=3, help="number of head modules")
parser.add_argument("--in_channel", type=int, default=64, help="dimension of channels")
parser.add_argument("--out_channel", type=int, default=64, help="dimension of channels")

# Training Parameters
parser.add_argument("--batch_size", type=int, default=256, help="number of batch")
parser.add_argument("--lr", type=float, default=1e-1, help="initial learning rate")
parser.add_argument("--wd", type=float, default=1e-4, help="initial learning rate")
parser.add_argument("--momentum", type=float, default=0.9, help="initial learning rate")
parser.add_argument("--epochs", type=int, default=100, help="number of max epochs")
parser.add_argument("--lrd", type=float, default=1e-1, help="multiplicative factor of learning rate decay")
parser.add_argument("--step_size", type=int, default=30, help="period of learning rate decay")

# wandb
parser.add_argument("--comment", type=str, default='model_params', help="save pth name")
args = parser.parse_args()


comment = args.comment
## Logger
config_logging(comment)
logging.info("arguments: {}".format(args))


## Dataloader
train_dataset, train_loader, valid_dataset, valid_loader, _, _ = prepare_dataloader(img_size=args.img_size, batch_size=args.batch_size)

print(len(train_loader))
print(len(valid_loader))


## model
model = build_fishnet(args)
model = model.to(device)
print(model)


## Train
total_train_losses = []
total_top1_train_accs = []
total_top5_train_accs = []
total_top1_valid_accs = []
total_top5_valid_accs = []

def train(model, device, train_loader, valid_loader):
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, 
                                weight_decay=args.wd, momentum=args.momentum)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.lrd)

    epochs = args.epochs
    saver = BestSaver(comment)

    for epoch in tqdm(range(1, epochs+1)):
        logging.info("Train Phase, Epoch: {}/{}".format(epoch, epochs))
        train_losses = AverageMeter()
        scheduler.step()
        
        # Train Phase
        model.train()

        total_top1_train_acc = 0
        total_top5_train_acc = 0
        for batch_num, data in enumerate(train_loader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
        
            # loss
            train_loss = criterion(output, labels)
            train_losses.update(train_loss.item(), images.shape[0])

            # acc
            _, pred = output.data.topk(5, 1, True, True)
            pred = pred.t()
            
            pred_labels = pred.eq(labels.view(1,-1).expand_as(pred))

            top1 = pred_labels[:1].reshape(-1).float().sum(0, keepdim=True)
            top5 = pred_labels[:5].reshape(-1).float().sum(0, keepdim=True)
            top1_acc = (top1/args.batch_size)*100
            top5_acc = (top5/args.batch_size)*100
            
            total_top1_train_acc += top1_acc
            total_top5_train_acc += top5_acc
            
            # backward
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            if batch_num % 20 == 0:
                logging.info(
                    "[{}/{}] # {}/{} loss: {:.4f}".format(
                        epoch, epochs, batch_num, len(train_loader), train_losses.val)
                )
        logging.info("Train_loss: {:.4f}, top1_train_acc: {:.4f}%, top5_train_acc: {:.4f}%".format(
            train_losses.avg, total_top1_train_acc.item()/len(train_loader), total_top5_train_acc.item()/len(train_loader)))
        total_train_losses.append(train_losses.avg)
        total_top1_train_accs.append(total_top1_train_acc.item()/len(train_loader))
        total_top5_train_accs.append(total_top5_train_acc.item()/len(train_loader))


        # Valid Phase
        model.eval()

        total_top1_valid_acc = 0
        total_top5_valid_acc = 0
        
        for batch_num, data in enumerate(valid_loader):
            images, labels = data

            images = images.to(device)
            labels = labels.to(device)
            
            with torch.no_grad():
                output = model(images)
            # acc
            _, pred = output.data.topk(5, 1, True, True)
            pred = pred.t()
            
            pred_labels = pred.eq(labels.view(1,-1).expand_as(pred))

            top1 = pred_labels[:1].reshape(-1).float().sum(0, keepdim=True)
            top5 = pred_labels[:5].reshape(-1).float().sum(0, keepdim=True)
            top1_acc = (top1/args.batch_size)*100
            top5_acc = (top5/args.batch_size)*100
            
            total_top1_valid_acc += top1_acc
            total_top5_valid_acc += top5_acc

        logging.info('top1_valid_acc: {:.4f}%, top5_valid_acc: {:.4f}'.format(
            total_top1_valid_acc.item()/len(valid_loader), total_top5_valid_acc.item()/len(valid_loader)))
        total_top1_valid_accs.append(total_top1_valid_acc.item()/len(valid_loader))
        total_top5_valid_accs.append(total_top5_valid_acc.item()/len(valid_loader))
        # save best model
        saver.save(total_top1_valid_acc.item()/len(valid_loader), model.state_dict())    

# run code
if __name__ == "__main__":
    train(model, device, train_loader, valid_loader)