import numpy as np
import pickle
import json
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description='download cifar-10 dataset & prepare train and test')

parser.add_argument('--t', default='train', type=str,
                    help='download data type(train/test)')
args = parser.parse_args()

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    
    if args.t == 'train':
        save_path = './train/'
    else:
        save_path = './test/'
    
    labels     = []
    file_names = []
    for key, data in dict.items():
        if key == b'labels':
            labels = data
        elif key == b'filenames':
            file_names = data
    
    img2labels_dict = {}
    for i in range(len(labels)):
        img2labels_dict[file_names[i].decode('utf-8')] = labels[i]
    
    for key, data in dict.items():
        if key == b'data':
            reshape_img = np.reshape(data, (10000,3,32,32))
            for i in range(len(reshape_img)):
                img = reshape_img[i]
                img = Image.fromarray(img.T, 'RGB')
                img = img.rotate(270, Image.NEAREST, expand=1)
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                img.save(save_path+'{}'.format(file_names[i].decode('utf-8')))

    return dict, img2labels_dict


file_path = './cifar-10-batches-py/'

img2labels_dic = {}
if args.t == 'train':
    for i in range(5):
        file = file_path+'data_batch_{}'.format(i+1)
        data, dic = unpickle(file)
        img2labels_dic.update(dic)

    with open('./train_img2labels.json', 'w') as f:
        json.dump(img2labels_dic, f, indent=4)

else:
    file = file_path+'test_batch'
    data, dic = unpickle(file)
    img2labels_dic.update(dic)

    with open('./test_img2labels.json', 'w') as f:
        json.dump(img2labels_dic, f, indent=4)
