


import numpy as np
import cv2
import os
import random

##将7200*6800的随机切割为256大小的图 每张大图切2000张小图，随机切割

def check_makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)
def cut_big_image_random(input_path,output_path):
    check_makedir(output_path)
    ###同时切割3种类型的图
    input_dir_list = ['tiff_npy','src','label','vis_label']
    output_dir_list = ['tiff_npy','src','label','vis_label']
    for name in output_dir_list:
        check_makedir(os.path.join(output_path,name))
    for filename in os.listdir(os.path.join(input_path,input_dir_list[0])):
        m=0
        # x=[]
        # y=[]
        if filename.endswith('.tiff.npy'):
            img_tiff=np.load(os.path.join(input_path,input_dir_list[0],filename))
            # img_tiff_show=img_tiff[:,:,0:3]
            # cv2.imshow("tiff show",img_tiff_show)
            img_src=cv2.imread(os.path.join(input_path,input_dir_list[1],filename.replace('.tiff.npy','.jpg')))
            # cv2.imshow('jpg show',img_src)
            # cv2.waitKey(0)
            img_label = cv2.imread(os.path.join(input_path, input_dir_list[2], filename.replace('.tiff.npy','_label.png')),cv2.IMREAD_GRAYSCALE)
            img_vis_label=cv2.imread(os.path.join(input_path, input_dir_list[3], filename.replace('.tiff.npy','.png')))

            for m in range(0,2000):
                i = random.randint(0,6800-256)
                j = random.randint(0,7200-256)
                # x.append(i)
                # y.append(j)
                path_save_tiff = os.path.join(output_path, input_dir_list[0],
                                              filename.replace('.tiff.npy', '_%04d_tiff.npy' % m))
                path_save_src = os.path.join(output_path, input_dir_list[1], filename.replace('.tiff.npy', '_%04d.jpg' % m))
                path_save_label = os.path.join(output_path, input_dir_list[2],
                                               filename.replace('.tiff.npy', '_label_%04d.png' % m))
                path_save_vis_label = os.path.join(output_path, input_dir_list[3],
                                                   filename.replace('.tiff.npy', '_%04d.png' % m))
                np.save(path_save_tiff, img_tiff[i:i + 256, j:j + 256, :])

                cv2.imwrite(path_save_src, img_src[i:i + 256, j:j + 256, :])
                cv2.imwrite(path_save_label, img_label[i:i + 256, j:j + 256])
                cv2.imwrite(path_save_vis_label, img_vis_label[i:i + 256, j:j + 256, :])
                m += 1
                print('finish figure_%04d' % m)
            print('finish all 1000 pictures of %r' % filename)
    print('finish cut all pictures')
##将7200*6800的大图切割为256大小的图 每张大图切728张小图，横竖切割
def cut_big_image(input_path,output_path):
    check_makedir(output_path)
    for filename in os.listdir(input_path):
        if filename.endswith('.jpg'):
            img = cv2.imread(os.path.join(input_path , filename))
            print(os.path.join(input_path , filename))
            m = 0
            for i in range(0, 26):
                for j in range(0, 28):
                    if m > 5824:
                        break
                    else:
                        m = m + 1
                    path = os.path.join(output_path , filename.replace('.jpg', '_%04d.jpg' % m))
                    cv2.imwrite(path, img[256 * i:256 * i + 256, 256 * j:256 * j + 256, :])
            print("done")
        if filename.endswith('_label.tif') :
            img=cv2.imread(os.path.join(input_path , filename))
            m = 0
            for i in range(0, 26):
                for j in range(0, 28):
                    if m > 5824:
                        break
                    else:
                        m = m + 1
                    path = os.path.join(output_path , filename.replace("MSS1 (2)","MSS1").replace("MSS2 (2)","MSS2").replace('_label.tif', '_%04d.png' % m))
                    cv2.imwrite(path, img[256 * i:256 * i + 256, 256 * j:256 * j + 256, :])
            print("done")

if __name__== "__main__":

    input_path=r'/media/aries/新加卷/seg_img/test_in/tiff_npy'
    # ##test_in文件夹要包含'tiff_npy','src','label','vis_label'这四个文件夹。
    # # 里面要包含要切割的图。且命名一样，后缀分别为".tiff.npy",".jpg","_label",".png"
    output_path=r'/media/aries/新加卷/seg_img/test_out'
    cut_big_image_random(input_path,output_path)