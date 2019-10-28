import cv2
import numpy as np
import os
import time

##将着色的label图转化为256*256大小的一维label，内部存储的是0-16


def encode(r, g, b):
    return (r * 256 + g) * 256 + b


def tif_to_npy(input_path, output_path):
    img = cv2.imread(input_path)  # input image 需要转为rgb
    # cv2.imshow("what", img)
    # cv2.waitKey()
    # cv2.destroyWindow("what")
    img = img[:, :,(2, 1, 0)]

    img2 = np.zeros((256, 256), dtype=np.uint8)  # 输出label图

    # tolist = [[0, 200, 0],
    #           [150, 250, 0],
    #           [150, 200, 150],
    #           [200, 0, 200],
    #           [150, 0, 250],
    #           [150, 150, 250],
    #           [250, 200, 0],
    #           [200, 200, 0],
    #           [200, 0, 0],
    #           [250, 0, 150],
    #           [200, 150, 150],
    #           [250, 150, 150],
    #           [0, 0, 200],
    #           [0, 150, 200],
    #           [0, 200, 250],
    #           [0, 0, 0]]
    tolist = [[0, 0, 0],
               [200, 0, 0],
               [0, 0, 200],
               [0, 100, 0],
               [0, 250, 0],
               [150, 250, 0],
               [255, 255, 255],
               [0, 150, 250]]
    labeltype = -1
    mydict = {}
    for i in tolist:
        labeltype += 1
        [r, g, b] = i
        # print(labeltype)
        if (mydict.get(encode(r, g, b)) != None): assert (0)
        mydict[encode(r, g, b)] = labeltype
    count = 0
    for x in np.nditer(img[:,:,0]):
        [r, g, b] = img[int(count / 256), int(count % 256)]
        img2[int(count / 256), int(count % 256)] = mydict[encode(r, g, b)]  # 输出在这里
        count += 1
    # cv2.imshow("t", img2)
    # cv2.waitKey()
    # cv2.destroyWindow("t")
    np.save(output_path,img2)

if __name__=="__main__":
    ts=time.time()
    input_dir=r'H:\pengqing\deeplabv3\data_8_12\train\vis_label'
    output_dir=r'H:\pengqing\deeplabv3\data_8_12\train\npy_generate'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    img_namelist = os.listdir(input_dir)
    for name in img_namelist:
        input_path = os.path.join(input_dir, name)
        output_path = os.path.join(output_dir, name.replace('.png', '.npy')).replace(".jpg", ".npy")
        tif_to_npy(input_path,output_path)
        print("finish figure %s" % name)
    print("finish all  %d figures" % len(img_namelist))
    print("tame elapse %d" % (time.time()-ts))

