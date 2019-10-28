# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
from tqdm import tqdm
class_name=["label_building","label_waterbody","label_grass",\
            "label_forest","label_field","label_road","label_bareland"]

# priority_ranking=[2,3,7,6,4,1,5]
priority_ranking=[6,5,1,2,4,7,3]


path=r"H:\pengqing\seg_img\test_fusion1"
save_path=r"H:\pengqing\seg_img\test_fusion1\fusion"
#使用说明：
#创建以上目录，并且保证只有图片，否则报错
#mask 最好为单通道 像素值只有0,1
#0代表 背景
#1代表 目标

## 读取图像，解决imread不能读取中文路径的问题
def cv_imread(filePath):
    cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    ## imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
    ##cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
    return cv_img

mm=set()
for i in range(len(class_name)):
    if not os.path.isdir(path+"\\"+class_name[i]):assert 0  #make sure to setup the /
    #correct document
    if not len(mm):
        mm.add(len(os.listdir(path+"\\"+class_name[i])))
    else:
        if len(os.listdir(path+"\\"+class_name[i]))  not in mm:
            assert 0 #make sure to generate the same amount of mask
file_no=mm.pop()
if not file_no:assert 0 #empty file

img_shape=cv_imread(path+"\\"+class_name[0]+"\\"+os.listdir(path+"\\"+class_name[0])[0]).shape
#dangerous usage may cause bug due to mask not in the same size
file_list=[]
r=img_shape[0]
c=img_shape[1]
for i in range(len(class_name)):
    file_list.append(os.listdir(path+"\\"+class_name[i]))
file_pointer=0
while file_pointer!=file_no:
    finnal_mask=np.zeros(img_shape,dtype=np.uint8)
    calc_mask=np.zeros((r,c,len(class_name)),dtype=np.uint8)
    for i in range(len(class_name)):
        img_tmp=cv_imread( path+"\\"+class_name[i]+"\\"+file_list[i][file_pointer])
        if len(img_tmp.shape)>=3:img_tmp=img_tmp[:,:,0]
        calc_mask[:,:,i]=img_tmp.copy()   #多层叠加
    count=0
    for x in tqdm(np.nditer(calc_mask[:,:,0])):  #
        chanel_val=np.sum(calc_mask[count//c][count%c])
        ro=count//c
        co=count%c
        bitint=np.zeros((1,len(priority_ranking)),dtype=np.uint8)
        channel_count=0
        final_pixel_val=0
        for i in priority_ranking:
            if calc_mask[ro][co][channel_count]==1:
                bitint[0,i-1]=1
            channel_count+=1
        prio_count=len(priority_ranking)
        for x in range(len(priority_ranking)):
            if bitint[0,prio_count-1]==1:
                break
            prio_count-=1
        if prio_count != 0:
            final_pixel_val = priority_ranking.index(prio_count)+1

        finnal_mask[ro][co]=final_pixel_val
        count+=1
    cv2.imencode('.png', finnal_mask)[1].tofile(save_path+"\\"+str(file_pointer)+".png")
    file_pointer+=1




