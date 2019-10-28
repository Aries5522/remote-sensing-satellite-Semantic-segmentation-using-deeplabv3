# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os

##将该文件给放入要转换的label的文件夹，修改路径然后直接运行
path=r"/media/aries/新加卷/deeplabv3/data_9_4/train/label"
odir_list=["label_building","label_waterbody","label_grass","label_forest","label_field","label_road","label_bareland"]
for name in odir_list:
    if os.path.isdir(path+"/"+name):continue
    os.mkdir(path+"/"+name)
for file in os.listdir(path):
    olddir=os.path.join(path,file)
    if os.path.isdir(olddir):continue
    if olddir.split(".")[-1]=='py':continue
    img=cv2.imread(file)
    r,c,_=img.shape
    img=img[:,:,0]
    imglist=[]
    for i in range(7):
        imglist.append(np.zeros((r,c),dtype=np.uint8))
    count=-1
    for x in np.nditer(img):
        count += 1
        assert(x>=0 and x<=7)   #label within range
        if x==0:continue
        imglist[x-1][count//c][count%c]=1

    for idx,str in enumerate(odir_list):
        save_path=path+"/"+odir_list[idx]+"/"+file
        cv2.imencode('.png', imglist[idx])[1].tofile(save_path)
