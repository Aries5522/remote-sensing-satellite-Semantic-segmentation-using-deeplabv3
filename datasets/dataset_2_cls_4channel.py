# -*- coding: utf-8 -*-
# @Time    : 2019/6/28 12:42
# @Author  : Aries
# @FileName: dataloader.py
# @Software: PyCharm
# @github  : Aries5522
"""train use 4channel data"""
import os
import torch
import numpy as np
import scipy.io as sio
import torch
from PIL import Image
import random
from torch.utils import data
from transforms import *
from joint_transforms import *
#
##num_class=2

palette=[0,0,0,
         200,0,0,
         0,0,200,
         0,100,0,
         0,250,0,
         150,250,0,
         255,255,255,
         0,150,250]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask    ###Image


torch.backends.cudnn.benchmark = True
ignore_label = 255
root = '/media/aries/新加卷/pengqing_projects_all/yaogan_2cls/data_9_4'

def make_dataset_2_cls(mode,cls):
    assert mode in ['train','val','test']
    assert cls in ['building','grass','waterbody','field','road','forest','bareland']
    items=[]
    if mode=='train' or mode=='val':
        image_path = os.path.join(root, 'train', 'tiff_npy')
        label_path=os.path.join(root,'train','label_%s' % cls)
        for img in os.listdir(image_path):
            item = (os.path.join(image_path, img),
                    os.path.join(label_path, img).replace("MSS1", "MSS1_label").replace("_tiff.npy",
                                                                                        ".png").replace(
                        "MSS2", "MSS2_label"),
                    )
            items.append(item)
            # pic = Image.open(
            #     os.path.join(label_path, img).replace("MSS1", "MSS1_label").replace("_tiff.npy", ".png").replace(
            #         "MSS2", "MSS2_label"))
            # pic = np.array(pic)
            # if np.sum(pic[pic == 1]) >= 256 * 256 * 0.00:
            #     item = (os.path.join(image_path, img),
            #             os.path.join(label_path, img).replace("MSS1", "MSS1_label").replace("_tiff.npy",
            #                                                                                 ".png").replace(
            #                 "MSS2", "MSS2_label"),
            #             )
            #     items.append(item)
        a = len(items)

        # random.shuffle(items)
        items_train = items[:int(a*10/13)]
        # items_train = items[0:3]
        items_val=items[int(a*10/13):]
        print(len(items_train))
        print(len(items_val))
        # items_val = items[int(0.9 * a):]
        return items_train, items_val
    # if mode== 'val':
    #     image_path = os.path.join(root, mode, 'tiff_npy')
    #     label_path = os.path.join(root, mode, 'label_%s' % cls)
    #     for img in os.listdir(image_path):
    #         item = (os.path.join(image_path, img),
    #                 os.path.join(label_path, img).replace("MSS1", "MSS1_label").replace("_tiff.npy", ".png").replace(
    #                     "MSS2", "MSS2_label"))
    #         items.append(item)
    #     return items,items
    if mode =='test':
        img_path = os.path.join(root, mode,'GF2_PMS2_E116.4_N39.0_20160827_L1A0001787565-MSS2') ##10
        img_name = os.listdir(img_path)
        for img in img_name:
            item = (os.path.join(img_path, img), img)
            items.append(item)
        return items,items  ####path list of imgs and labels

class yaogan(data.Dataset):
    def __init__(self, mode,cls, input_transform=None, sliding_crop=None,
               joint_transform=None, target_transform=None):
        self.mode = mode
        self.cls=cls
        self.imgs_train,self.imgs_val = make_dataset_2_cls(mode,cls)
        self.input_transform = input_transform
        self.joint_transform = joint_transform
        self.sliding_crop = sliding_crop
        self.target_transform = target_transform

        if (len(self.imgs_train)) == 0:
            print("found 0 images, please check dataset")

    def __getitem__(self, index):
        if self.mode == 'test':
            img_path, img_name = self.imgs_train[index]
            img = np.load(img_path)
            if self.input_transform is not None:
                img = self.input_transform(img)
            return img, img_name

        if self.mode == 'train':
            img_path, mask_path= self.imgs_train[index]
            img = np.load(img_path)
            mask = Image.open(mask_path)
            # Image._show(img)
            # Image._show(colorize_mask(np.array(mask)))
        if self.mode=='val':
            img_path,mask_path=self.imgs_val[index]
            img=np.load(img_path)
            mask=Image.open(mask_path)
        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)
            # Image._show(img)
            # Image._show(colorize_mask(np.array(mask)))
        # if self.sliding_crop is not None:
        #     img_slices, mask_slices, slices_info = self.sliding_crop(img, mask)
        #     if self.input_transform is not None:
        #         img_slices = [self.input_transform(e) for e in img_slices]
        #     if self.target_transform is not None:
        #         mask_slices = [self.target_transform(e) for e in mask_slices]
        #     img, mask = torch.stack(img_slices, 0), torch.stack(mask_slices, 0)
        #     return img, mask, torch.LongTensor(slices_info)
        if self.input_transform is not None:
            img = self.input_transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        return img, mask  ##img_4channel mask_1channel

    def __len__(self):
        if self.mode=='train':
            return len(self.imgs_train)
        if self.mode=='val':
            return  len(self.imgs_val)
        return len(self.imgs_train)

if __name__ == "__main__":
    yaogan = yaogan(mode='train', cls='bareland',joint_transform=None, input_transform=None, target_transform=None)
    import cv2
    img, mask, = yaogan.__getitem__(5)
    cv2.imshow('what',img[:,:,0:3])
    cv2.waitKey()
    cv2.destroyWindow('what')
    img=Image.fromarray(img[:,:,0:3]).convert('RGB')
    Image._show(img)

    Image._show(mask)
    mask=np.array(mask)
    mask_pic=colorize_mask(mask)
    Image._show(mask_pic)
    print(mask)
    print("show 1 figure")



