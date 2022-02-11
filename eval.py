import argparse

import numpy as np
import torch

from models.fishnet import build_fishnet
from utils import prepare_dataloader

parser = argparse.ArgumentParser(description='FishNet')

parser.add_argument("--img_size", type=int, default=32, help="img_size")

# Model Parameters
parser.add_argument("--n_tail", type=int, default=3, help="number of tail modules")
parser.add_argument("--n_body", type=int, default=3, help="number of body modules")
parser.add_argument("--n_head", type=int, default=3, help="number of head modules")
parser.add_argument("--in_channel", type=int, default=64, help="dimension of channels")
parser.add_argument("--out_channel", type=int, default=64, help="dimension of channels")

args = parser.parse_args()
print(args)

## Device
device = torch.device("cuda:0")
print(device)


## Dataloader
_,_, _, _, test_dataset, test_loader = prepare_dataloader(img_size=args.img_size)

print('length of batch size, test data: {}'.format(len(test_loader)))

## model
model = build_fishnet(args)
model = model.to(device)


## load params
model.load_state_dict(torch.load('./utils_model.pth'))


def test(model, device, test_loader):

    # Test Phase
    model.eval()

    correct = 0
    count   = 0
    for batch_num, data in enumerate(test_loader):
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            output = model(images)
        pred_labels = np.argmax(output.cpu().detach().numpy())

        if labels == pred_labels:
            correct += 1
            count   += 1
        else:
            count += 1
        if batch_num % 250 == 0:
            print('{}/{}, test_acc: {:.4f}%'.format(batch_num, len(test_loader),(correct/count)*100))
    
    print('acc: {:.4f}%'.format((correct/count)*100))


if __name__ == "__main__":
    test(model, device, test_loader)