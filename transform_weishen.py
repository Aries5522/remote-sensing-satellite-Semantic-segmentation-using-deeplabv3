import gdal,os
from matplotlib import pyplot as plt
import numpy as np
import cv2

def tif_weishen_transform1(img_dir):
    print(img_dir)
    data_gdal=gdal.Open(img_dir)
    print(data_gdal)
    # proj = data_gdal.GDALGetProjectionRef()
    im_width=data_gdal.RasterXSize
    im_height=data_gdal.RasterYSize
    im_bands=data_gdal.RasterCount
    im_data=data_gdal.ReadAsArray(0,0,im_width,im_height)
    tmp_pic=np.zeros([im_height,im_width,im_bands])
    
    tmp_pic[:,:,0]=im_data[0,0:im_height,0:im_width]
    tmp_pic[:,:,1]=im_data[1,0:im_height,0:im_width]
    tmp_pic[:,:,2]=im_data[2,0:im_height,0:im_width]    
    tmp_pic[:,:,3]=im_data[3,0:im_height,0:im_width]    
    
    channel_0=tmp_pic[:,:,0]
    channel_1=tmp_pic[:,:,1]
    channel_2=tmp_pic[:,:,2]
    channel_3=tmp_pic[:,:,3]
    # channel_0_max=np.max(channel_0)
    # channel_1_max=np.max(channel_1)
    # channel_2_max=np.max(channel_2)
    # channel_3_max=np.max(channel_3)
    final_out=tmp_pic.copy()

    ######################
    to_sort=np.array(channel_0.reshape((im_height*im_width,1)))

    sorted_data=np.sort(to_sort,axis=0)
    all_pix_num=im_height*im_width
    up_percent=2
    dowm_percent=2
    up_yuzhi=sorted_data[int(all_pix_num*(1-up_percent/100))]
                         
                         
    down_yuzhi=sorted_data[int(all_pix_num*(dowm_percent/100))]
    #down_yuzhi=0
    #print(up_yuzhi,down_yuzhi)

    output_img=np.where(channel_0>=up_yuzhi,up_yuzhi,channel_0)
    output_img=np.where(channel_0<=down_yuzhi,0,output_img)
    
    output_img=(output_img-down_yuzhi)/(up_yuzhi-down_yuzhi)*255
    final_out[:,:,0]=output_img
    ######################
    ######################
    to_sort=np.array(channel_1.reshape((im_height*im_width,1)))

    sorted_data=np.sort(to_sort,axis=0)
    all_pix_num=im_height*im_width
   
    up_yuzhi=sorted_data[int(all_pix_num*(1-up_percent/100))]

    down_yuzhi=sorted_data[int(all_pix_num*(dowm_percent/100))]
    #down_yuzhi=0
    #print(up_yuzhi,down_yuzhi)
    output_img=np.where(channel_1>=up_yuzhi,up_yuzhi,channel_1)
    output_img=np.where(output_img<=down_yuzhi,0,output_img)
    
    output_img=(output_img-down_yuzhi)/(up_yuzhi-down_yuzhi)*255
    final_out[:,:,1]=output_img
    ######################
    ######################
    to_sort=np.array(channel_2.reshape((im_height*im_width,1)))

    sorted_data=np.sort(to_sort,axis=0)
    all_pix_num=im_height*im_width
   
    up_yuzhi=sorted_data[int(all_pix_num*(1-up_percent/100))]
                         
                         
    down_yuzhi=sorted_data[int(all_pix_num*(dowm_percent/100))]
    #down_yuzhi=0
    #print(up_yuzhi,down_yuzhi)
    output_img=np.where(channel_2>=up_yuzhi,up_yuzhi,channel_2)
    output_img=np.where(output_img<=down_yuzhi,0,output_img)
    
    output_img=(output_img-down_yuzhi)/(up_yuzhi-down_yuzhi)*255
    final_out[:,:,2]=output_img
    ######################
	######################
    to_sort=np.array(channel_3.reshape((im_height*im_width,1)))

    sorted_data=np.sort(to_sort,axis=0)
    all_pix_num=im_height*im_width
  
    up_yuzhi=sorted_data[int(all_pix_num*(1-up_percent/100))]
                         
                         
    down_yuzhi=sorted_data[int(all_pix_num*(dowm_percent/100))]
    #down_yuzhi=0
    #print(up_yuzhi,down_yuzhi)
    output_img=np.where(channel_3>=up_yuzhi,up_yuzhi,channel_3)
    output_img=np.where(output_img<=down_yuzhi,0,output_img)
    
    output_img=(output_img-down_yuzhi)/(up_yuzhi-down_yuzhi)*255
    final_out[:,:,3]=output_img
    ######################
	######################


    # output_img[:,:,0]=cv2.equalizeHist(output_img[:,:,0])
    # output_img[:,:,1]=cv2.equalizeHist(output_img[:,:,1])
    # output_img[:,:,2]=cv2.equalizeHist(output_img[:,:,2])

    #print(output_img)
    # cv2.imwrite(img_dir.replace('.tif','').replace('rssrai2019_change_detection',
                                                   # 'rssrai2019_change_detection_processed')+'.jpg',final_out[:,:,0:3])
    '''
    以上的代码做到了：1.位深由16——>8
                     2.由4通道B,G,R,NIR变为3通道R,G,B
    '''
    # cv2.imwrite(img_dir.replace('_CUT','_CUT_transform').replace('tiff','jpg'),final_out[:,:,0:3])
    # cv2.imwrite(img_dir.replace('.tiff','_8wei.tiff'),final_out[:,:,0:4])
    
    ##原本是有下面的代码，但暂时不知道有什么用处，就先注释了
    imgname = img_dir.split('/')[-1]
    out_path = '/media/aries/新加卷/data_all/seg_img/data_all/test/'+ imgname
    cv2.imwrite(out_path.replace('tiff', 'jpg'), final_out[:, :, 0:3])
    driver = gdal.GetDriverByName('GTiff').Create(out_path, final_out.shape[1], final_out.shape[0], 4, gdal.GDT_Byte)
    driver.GetRasterBand(1).WriteArray(final_out[:, :, 0])
    driver.GetRasterBand(2).WriteArray(final_out[:, :, 1])
    driver.GetRasterBand(3).WriteArray(final_out[:, :, 2])
    driver.GetRasterBand(4).WriteArray(final_out[:, :, 3])

    del driver


if __name__== '__main__':
    tif_path ='/media/aries/新加卷/data_all/seg_img/data_all/原始影像_16位编码_4通道'
    dirlist=os.listdir(tif_path)
    for filename in dirlist:
        print('process ', filename)
        filename = os.path.join(tif_path,filename)
        tif_weishen_transform1(filename)
