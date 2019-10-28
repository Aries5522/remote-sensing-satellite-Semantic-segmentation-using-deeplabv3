# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
class_name=["label_building","label_waterbody","label_grass",\
            "label_forest","label_field","label_road","label_bareland"]
path="D:\\WORK\\UESTC\\54所项目\\杂项\\融合"
save_path="D:\\WORK\\UESTC\\54所项目\\杂项\\融合\\Fusion"
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
        calc_mask[:,:,i]=img_tmp.copy()
    count=0
    for x in np.nditer(calc_mask[:,:,0]):
        chanel_val=np.sum(calc_mask[count//c][count%c])
        final_pixel_val=-1
        if chanel_val>=2:
            suc=0
        else:
            suc=1
            if chanel_val==0:
                final_pixel_val=0
            else:
                final_pixel_val=np.where(calc_mask[count//c][count%c])[0][0]+1
        if not suc:
            ro=count//c
            co=count%c
            inner_count=0
            while 1:
                inner_count+=1
                if inner_count>=10:assert 0 #"not possible to iterate more than 10 times"
                mm={}
                for rcount in list(range(-inner_count,inner_count)):
                    for ccount in list(range(-inner_count,inner_count)):
                        nr=ro+rcount
                        nc=co+ccount
                        if nr<0 or nr>=r or nc<0 or nc>=c:continue
                        if np.sum(calc_mask[nr][nc])>=2:continue
                        else:
                            if not len(np.where(calc_mask[nr][nc])[0]):
                                if mm.get(0)==None:
                                    mm[0]=1
                                else:
                                    mm[0]+=1
                            else:
                                cls=np.where(calc_mask[nr][nc])[0][0]+1
                                if mm.get(cls)==None:
                                    mm[cls]=1
                                else:
                                    mm[cls]+=1
                if not len(mm):continue
                sort_list=[]
                for key in mm:
                    sort_list.append((mm[key],key))
                sort_list.sort()

                if len(sort_list)==1:
                    finnal_mask[count//c][count%c]=sort_list[0][1]
                    break
                else:
                    if sort_list[-1][0]==sort_list[-2][0]:
                        continue
                    else:
                        finnal_mask[count//c][count%c]=sort_list[-1][1]
                        break

        else:
            finnal_mask[count//c][count%c]=final_pixel_val
        count+=1
    cv2.imencode('.png', finnal_mask)[1].tofile(save_path+"\\"+str(file_pointer)+".png")
    file_pointer+=1




