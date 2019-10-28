# remote-sensing-satellite-Semantic-segmentation-using-deeplabv3

项目整体是基于遥感比赛数据集整理的：
整个项目架构如下：
--datasets

----dataset（first you neet to change the root to you own path）

--model

----aspp.py

----deeplabv3.py

----resnet.py

----resnet18.py

--train

----train

----vis（主要用来测试上色的一个小脚本）

----test

----train（分别包含label，npy，src三个文件夹）

----val（分别包含label，npy，src三个文件夹）

--script(存放用来做图像预处理的一些脚本)

开始试验的时候，确认npy和src对应好之后就能运行train了，最终的模型和中途预测结果都在ckpt下.如果你需要数据的话，可以邮件联系我airespengqing@gmail.com


数据存放方式：
！[](./images/data.png)


下面是一些训练结果
！[](./images/field.PNG)

！[](./images/road.PNG)

！[](./images/waterbody.PNG)

！[src](./images/out15.jpg)

！[](./images/pred_img_building.png)

！[](,/../images/fusion_out_15.png)