import cv2
import os
import numpy as np
def check_makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def img_seg(img_path, output_path):
    check_makedir(output_path)
    img_namelist = os.listdir(img_path)
    for name in img_namelist:
        img = cv2.imread(os.path.join(img_path, name))
        # img=np.load(os.path.join(img_path, name))
        h, w, _ = img.shape
        assert w == 7200 and h == 6800
        img_outputdir = os.path.join(output_path, name.replace('.jpg', ''))
        if not os.path.exists(img_outputdir):
            os.makedirs(img_outputdir)

        index = 0
        for i in range(10):
            for j in range(10):
                output_name = os.path.join(img_outputdir, name.replace('.jpg', '_%.2d.jpg') % int(index))
                cv2.imwrite(output_name, img[i * 680:(i * 680 + 680), j * 720:(j * 720 + 720), :])
                # np.save(output_name, img[i * 680:(i * 680 + 680), j * 720:(j * 720 + 720), :])
                index = index + 1
        print("finish picture %s" % name)

if __name__ == '__main__':
    img_path = r'H:\pengqing\seg_img\data_all\3_channels_xiongan_big'
    output_path = r'H:\pengqing\seg_img\data_all\3_channels_xiongan_small1'
    img_seg(img_path, output_path)
#
# import scipy.misc
# import numpy as np
# import os
# import cv2
#
# class ImageProcess():
#     def __init__(self,path_image,output_path):
#         self.path_image = path_image
#         self.output_path = output_path
#     def tif_to_jpg(self):
#         for filename in os.listdir(self.path_image):
#           if filename.endswith('.tif'):
#             img = cv2.imread(self.path_image+filename)
#             print(self.path_image+filename)
#             img = img[:,:,1:4]
#             cv2.imwrite(self.output_path+filename.replace('.tif','')+".jpg", img)
#             print("done")
#     def cut_big_image(self):
#         for filename in os.listdir(self.path_image):
#           if filename.endswith('label.tif'):
#             img = cv2.imread(self.path_image+filename)
#             print(self.path_image+filename)
#             m=0
#             for i in range(0, 26):
#                 for j in range(0, 28):
#                     if m > 5824:
#                         break
#                     else:
#                         m = m + 1
#                     path = self.output_path+filename.replace('.tif','_%d.tif' % m)
#                     cv2.imwrite(path, img[256 * i:256 * i + 256,  256 * j:256 * j + 256, :])
#             print("done")
#
# if __name__ == '__main__':
#     path_image = "H:\pengqing\img"
#     output_path = "H:\pengqing\img_lable"
#     imageprocess1 = ImageProcess(path_image,output_path)
#     imageprocess1.tif_to_jpg()
#     imageprocess1.cut_big_image()
#     img = io.imread('F:/DataSet/遥感语义分割竞赛/rssrai2019_semantic_segmentation/train/GF2_PMS1__20160827_L1A0001793003-MSS1 (2).tif')
#     img = img[:,:,0:4]
#     print(img[1:10,1:10,:])
#     io.imsave('F:/Semantic-Segmentation/Segmentation-Competition/Dataset/src/1.tif', img)
#     print("end")
