import torch.utils.data as data
from PIL import Image
import numpy as np

###############################################################################

class Dataset(data.Dataset):

    def __init__(self, data, label,
                 transform=None,target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.data = data
        self.labels = label

    def __getitem__(self, index):
       
        img, target = self.data[index], self.labels[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # print(img.shape)
        if img.shape[0] != 1:
            #print(img)
            img = Image.fromarray(
              np.uint8(np.asarray(img.transpose((1, 2, 0)))) )
        #
        elif img.shape[0] == 1:
            im = np.uint8(np.asarray(img))
            # print(np.vstack([im,im,im]).shape)
            im = np.vstack([im, im, im]).transpose((1, 2, 0))
            img = Image.fromarray(im)

        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.transform is not None:
            img = self.transform(img)
            #  return img, target
        return img, target
    def __len__(self):
        return len(self.data)
