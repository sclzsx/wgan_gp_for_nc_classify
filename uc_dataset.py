import os
import cv2
from PIL import Image
import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
import random
import shutil

class_names = [i.name for i in Path('../UCMerced_LandUse/Images').iterdir() if i.is_dir()]
class_ids = [i for i in range(len(class_names))]
class_name2id = dict(zip(class_names, class_ids))
class_id2name = dict(zip(class_ids, class_names))

def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def split_ucdataset_for_wgangp_and_classify(dataset_dir, class_names=None, train_num=70, image_size=64):
    if class_names is not None:
        paths = [i for i in Path(dataset_dir).rglob('*.*') if i.name[:-6] in class_names]
    else:
        paths = [i for i in Path(dataset_dir).rglob('*.*')]

    train_dir = str(Path(dataset_dir).parent) + '/train' + str(image_size)
    test_dir = str(Path(dataset_dir).parent) + '/test' + str(image_size)
    mkdir(train_dir)
    mkdir(test_dir)

    train_ids = [i for i in range(train_num)]

    for path in paths:
        image = Image.open(str(path)).convert('RGB')
        image = image.resize((image_size, image_size), Image.ANTIALIAS)
        id = int(path.name[-6:-4])
        if id in train_ids:
            image.save(train_dir + '/' + path.name[:-3] + 'jpg', quality=100)
        else:
            image.save(test_dir + '/' + path.name[:-3] + 'jpg', quality=100)
        
    print('done')



class UCDataset(Dataset):
    def __init__(self, image_dir, image_size=256):
        self.paths = [i for i in Path(image_dir).rglob('*.*')]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
                        
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        image = Image.open(str(path))
        img_tensor = self.transform(image)
        
        class_name = path.name[:-6]
        class_id = class_name2id[class_name]

        label_tensor = torch.from_numpy(np.ascontiguousarray(class_id).astype('int64')).squeeze()

        return img_tensor, label_tensor


if __name__ == '__main__':
    # split_ucdataset_for_wgangp_and_classify('../UCMerced_LandUse/Images', class_names=class_names)

    dataset = UCDataset('../UCMerced_LandUse/train', image_size=64)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    for i, batch_data in enumerate(loader):
        print(batch_data[0].shape, batch_data[1])
        if i % 10 == 0 and i > 0:
            print('Check done', i)
            break