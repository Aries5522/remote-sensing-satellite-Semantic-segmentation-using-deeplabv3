# remote-sensing-satellite-Semantic-segmentation-using-deeplabv3

﻿项目整体是基于遥感比赛数据集整理的：
整个项目架构如下：
--datasets
----dataset（修改数据路径）
----data_process_fast(将label变为npy存下来)
--model
----aspp.py
----deeplabv3.py
----resnet.py
----resnet18.py
--train
----train
----vis（主要用来测试上色的一个小脚本）
--rssrai2019（存放数据集）
----test
----train（分别包含label，npy，src三个文件夹）
----val（分别包含label，npy，src三个文件夹）
--script(存放用来做图像预处理的一些脚本)

开始试验的时候，确认npy和src对应好之后就能运行train了，最终的模型和中途预测结果都在ckpt下


