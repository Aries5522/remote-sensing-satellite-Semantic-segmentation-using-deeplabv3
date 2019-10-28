# remote-sensing-satellite-Semantic-segmentation-using-deeplabv3

项目整体是基于遥感的一个项目数据集整理的：
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


下面是一些训练结果存放在images里面，可以大概查看下。

|分类类别|模型大小|像素精度|iou|
|-----|--------|------|-----|
|建筑（building）|	58.5m	|0.9522|	0.8811|
|水体（water-body）|	|0.8980	|0.8090|
|草地（grass）	||0.6654	|0.4906（过拟合严重）|
|森林（forest）||	0.5817	|0.4578（过拟合严重）|
|道路（road）	||0.5606	|0.3560（过拟合严重）|
|耕地（field）	||0.9026	|0.8566|
|裸地（bare-land）	||0.4884	|0.3698（数据误标）|