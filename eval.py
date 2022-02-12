import argparse

import torch
from tqdm import tqdm
from models.fishnet import build_fishnet
from utils import prepare_dataloader


## Device
device = torch.device("cuda:0")
print(device)


parser = argparse.ArgumentParser(description='FishNet')
parser.add_argument("--img_size", type=int, default=32, help="img_size")

# Model Parameters
parser.add_argument("--n_tail", type=int, default=3, help="number of tail modules")
parser.add_argument("--n_body", type=int, default=3, help="number of body modules")
parser.add_argument("--n_head", type=int, default=3, help="number of head modules")
parser.add_argument("--in_channel", type=int, default=64, help="dimension of channels")
parser.add_argument("--out_channel", type=int, default=64, help="dimension of channels")

# Training Parameters
parser.add_argument("--batch_size", type=int, default=1, help="number of batch")


args = parser.parse_args()
print(args)

## Dataloader
_,_, _, _, test_dataset, test_loader = prepare_dataloader(
    batch_size=args.batch_size,
    img_size=args.img_size)

print(len(test_loader))


## Model
model = build_fishnet(args)
model = model.to(device)
print(model)


## load model parameter
model.load_state_dict(torch.load('./utils_model2.pth'))


# test
def test(model, device, test_loader):
    
    model.eval()

    total_top1_test_acc = 0
    total_top5_test_acc = 0
    
    for batch_num, data in tqdm(enumerate(test_loader)):
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
        
        total_top1_test_acc += top1_acc
        total_top5_test_acc += top5_acc
    
    
    print("top 1 accuracy: {:.4f}%, top 5 accuracy: {:.4f}%".format(
        total_top1_test_acc.item()/len(test_loader), total_top5_test_acc.item()/len(test_loader)))


# run code
if __name__ == "__main__":
    test(model, device, test_loader)