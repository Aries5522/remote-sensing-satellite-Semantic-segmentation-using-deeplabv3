from utils import evaluate
import cv2


gt_path=r"C:\Users\lenovo\Desktop\data14\label\GF2_PMS2_E116.2_N39.1_20180322_L1A0003077112-MSS2_label.png"
fusion_path=r"H:\pengqing\seg_img\test_fusion1\fusion\0.png"
gt=cv2.imread(gt_path)
pred=cv2.imread(fusion_path)

gt_all=[]
pre_all=[]
gt_all.append(gt[:,:,0])
pre_all.append(pred[:,:,0])
acc, acc_cls, mean_iu, fwavacc, single_cls_acc, kappa,iu=evaluate(pre_all,gt_all,8)
print(iu)
print(single_cls_acc)
print('[acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f],[kappa %.5f]' % (
         acc, acc_cls, mean_iu, fwavacc,kappa))
