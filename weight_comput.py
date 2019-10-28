import numpy as np
from datasets.dataset import yaogan
from torch.utils.data import DataLoader
import transforms

####计算weight权重

target_transform= transforms.MaskToTensor()
input_transfom=transforms.MaskToTensor()

if __name__ =="__main__":
    yaogan=yaogan(mode='train',input_transform=input_transfom,target_transform=target_transform)
    train_loader=DataLoader(yaogan,batch_size=24,num_workers=8)
    weight=np.zeros((1,8)).squeeze(0).astype(np.uint)
    for index,data in enumerate(train_loader):
        img,label=data
        label=np.array(label).reshape(256*256*24,1).squeeze(1).astype(np.uint8)
        weight_=np.bincount(label,minlength=8)
        weight=weight_+weight
    print(weight)
    pix_partion=weight/weight.sum()  ##计算像素占比
    weight_all=1/pix_partion  ##计算权重
    max=max(weight_all)
    weight_all=weight_all/max  ##权重归一
    print(weight_all)
    print(pix_partion)

# import os
# from PIL import Image
#
# img_path='/home/aries/work/遥感语义分割竞赛/yaoganbisai/rssrai2019_data/val/lable'
# list=os.listdir(img_path)
# for img in list:
#     src = os.path.join(os.path.abspath(img_path), img)
#     dst = os.path.join(os.path.abspath(img_path), img.replace('.tif','.jpg'))
#     os.rename(src,dst)




