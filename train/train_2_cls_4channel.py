import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.utils as vutils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torchvision.transforms as standard_transforms
from tensorboardX import SummaryWriter
from torch.backends import cudnn
from model.deeplabv3 import DeepLabV3

from datasets.dataset_2_cls_4channel import yaogan

from utils import check_mkdir, evaluate, AverageMeter, CrossEntropyLoss2d, FocalLoss2d
import transforms as extended_transforms
import joint_transforms
from collections import OrderedDict
import time
import random
import os
import numpy as np

cudnn.benchmark = True

args = {
    'num_class': 2,
    'batch_size':64,
    'epoch_num': 90,
    'ckpt_path': '../ckpt',
    'exp':'exp',
    'exp_name': 'train_segnet_road_',
    'training_cls':'road',
    'lr': 1,
    'print_freq': 20,
    'use_gpu': torch.cuda.is_available(),
    'step_size': 3,
    'num_works': 8,
    'val_batch_size': 1,
    'val_save_to_img_file': True,
    'val_img_sample_rate': 0.05,
    'snapshot': '',
    'weight':''
}

"""###############------main--------###############"""

def main(train_args):
    check_mkdir(os.path.join(train_args['ckpt_path'], args['exp']))
    check_mkdir(os.path.join(train_args['ckpt_path'], args['exp'], train_args['exp_name']))
    model=DeepLabV3('1')

    # print(model)
    device=torch.device("cuda")

    num_gpu = list(range(torch.cuda.device_count()))
    """###############------use gpu--------###############"""
    if args['use_gpu']:
        ts = time.time()
        print(torch.cuda.current_device())
        print(torch.cuda.get_device_name(0))

        model = nn.DataParallel(model, device_ids=num_gpu)
        model=model.to(device)
        print("Finish cuda loading ,time elapsed {}", format(time.time() - ts))
    else:
        print("please check your gpu device,start training on cpu")
    """###############-------中间开始训练--------###############"""
    if len(train_args['snapshot']) == 0:
        curr_epoch = 1
        train_args['best_record'] = {'epoch': 0, 'val_loss': 1e10, 'acc': 0, 'acc_cls': 0, 'mean_iu': 0, 'fwavacc': 0}
        # model.apply(weights_init)
    else:
        print("train resume from " + train_args['snapshot'])

        state_dict = torch.load(os.path.join(train_args['ckpt_path'],args['exp'],
                                             train_args['exp_name'], train_args['snapshot']))
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        # model.load_state_dict(
        #     torch.load(os.path.join(train_args['ckpt_path'],args['exp'],train_args['exp_name'], train_args['snapshot'])))

        split_snapshot = train_args['snapshot'].split('_')

        curr_epoch = int(split_snapshot[1]) + 1
        train_args['best_record'] = {'epoch': int(split_snapshot[1]), 'val_loss': float(split_snapshot[3]),
                                     'acc': float(split_snapshot[5]), 'acc_cls': float(split_snapshot[7]),
                                     'mean_iu': float(split_snapshot[9]),
                                     'fwavacc': float(split_snapshot[11])}

    model.train()

    mean_std = ([0.485, 0.456, 0.406,0.450], [0.229, 0.224, 0.225,0.225])
    """#################---数据增强和数据变换等操作------########"""
    input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])  ##Nomorlized
    target_transform = extended_transforms.MaskToTensor() # target  to tensor

    joint_transform=joint_transforms.Compose(
        [joint_transforms.RandomHorizontallyFlip(),
         joint_transforms.RandomCrop((256,256),padding=0),
         joint_transforms.Rotate(degree=90)]
    )###data_augment

    restore = standard_transforms.Compose([
        extended_transforms.DeNormalize(*mean_std),
        extended_transforms.channel_4_to_channel_3(4,3), ##默认3通道如果四通道会转成三通道
        standard_transforms.ToPILImage(),
    ])  # DeNomorlized，出来是pil图片了

    visualize = standard_transforms.Compose([
        standard_transforms.Resize(256),
        standard_transforms.CenterCrop(256), ##中心裁剪，此处可以删除
        standard_transforms.ToTensor()
    ])  # resize 大小之后转tensor
    """#################---数据加载------########"""
    train_set = yaogan(mode='train', cls=train_args['training_cls'],joint_transform=None,
                       input_transform=input_transform, target_transform=target_transform)
    train_loader = DataLoader(train_set, batch_size=train_args['batch_size'],
                              num_workers=train_args['num_works'], shuffle=True)
    val_set = yaogan(mode='val', cls=train_args['training_cls'],
                     input_transform=input_transform, target_transform=target_transform)
    val_loader = DataLoader(val_set, batch_size=1,
                            num_workers=train_args['num_works'], shuffle=False)

    # test_set=yaogan(mode='test',cls=train_args['training_cls'],joint_transform=None,
    #                 input_transform=input_transform,target_transform=None)
    # test_loader=DataLoader(test_set,batch_size=1,
    #                        num_workers=train_args['num_works'], shuffle=False)

    optimizer = optim.Adadelta(model.parameters(), lr=train_args['lr'])

    ##define a weighted loss (0weight for 0 label)
    # weight=[0.09287939 ,0.02091968 ,0.02453979, 0.25752962 ,0.33731845, 1.,
    #         0.09518322, 0.52794035 ,0.24298112 ,0.02657369, 0.15057124 ,0.36864611,
    #         0.25835161,0.16672758 ,0.40728756 ,0.00751281]
    """###############-------训练数据权重--------###############"""
    if train_args['weight'] is not None:
        weight=[0.1, 1.]
        weight=torch.Tensor(weight)
    else:
        weight=None
    criterion = nn.CrossEntropyLoss(weight=weight,reduction='elementwise_mean', ignore_index=-100).to(device)
    # criterion=nn.BCELoss(weight=weight,reduction='elementwise_mean').cuda()

    check_mkdir(train_args['ckpt_path'])
    check_mkdir(os.path.join(train_args['ckpt_path'], args['exp']))
    check_mkdir(os.path.join(train_args['ckpt_path'], args['exp'], train_args['exp_name']))
    open(os.path.join(train_args['ckpt_path'], args['exp'], train_args['exp_name'], str(time.time()) + '.txt'), 'w').write(
        str(train_args) + '\n\n')
    """###############-------start training--------###############"""
    for epoch in range(curr_epoch, train_args['epoch_num'] + 1):

        adjust_lr(optimizer, epoch)
        train(train_loader, model, criterion, optimizer, epoch, train_args,device)
        val_loss = validate(val_loader, model, criterion, optimizer, restore,
                            epoch, train_args, visualize,device)
    writer.close()

def train(train_loader, model, criterion, optimizer, epoch, train_args,device):
    t1=time.time()
    train_loss = AverageMeter()
    curr_iter = (epoch - 1) * len(train_loader)
    gts_all, predictions_all =[], []
    for i, data in enumerate(train_loader):

        inputs, labels = data##inputs shape(B,C,H,W),label shape(B,H,W)

        assert inputs.size()[2:] == labels.size()[1:]
        N = inputs.size(0)
        inputs = Variable(inputs)
        labels = Variable(labels)
        if train_args['use_gpu']:
            inputs = Variable(inputs).to(device)
            labels = Variable(labels).to(device)
        optimizer.zero_grad()


        outputs = model(inputs)  ##outputs shape(B,num_class,H,W)
        predictions = outputs.data[:].max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()
        assert outputs.size()[2:] == labels.size()[1:]
        assert outputs.size()[1] == train_args['num_class']
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss.update(loss.data, N)

        curr_iter += 1
        writer.add_scalar('train_loss', train_loss.avg, curr_iter)

        if (i + 1) % train_args['print_freq'] == 0:
            print('[epoch %d], [iter %d / %d], [train loss %.7f]' % (
                epoch, i + 1, len(train_loader), train_loss.avg
            ))

        if random.random() >0.8:
            gts_all.append(labels.data[:].squeeze_(0).cpu().numpy())
            predictions_all.append(predictions)

    acc, acc_cls, mean_iu, fwavacc,single_cls_acc ,kappa= evaluate(predictions_all, gts_all, train_args['num_class'])
    print('Finish training [epoch %d],'
          ' [train loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f],[kappa %.5f]' % (
              epoch, train_loss.avg, acc, acc_cls, mean_iu, fwavacc,kappa))
    print('training_single_cls_acc:')
    print(single_cls_acc)

    t2=time.time()
    print('Finish training [epoch %d],time elapsed %d min %d second'
          % (epoch,int((t2-t1)/60),int((t2-t1)%60)))

def validate(val_loader, model, criterion, optimizer, restore, epoch, train_args, visualize,device):
    model.eval()
    val_loss = AverageMeter()
    inputs_all, gts_all, predictions_all = [], [], []

    for vi, data in enumerate(val_loader):

        inputs, gts, = data

        N = inputs.size(0)
        inputs = Variable(inputs)
        gts = Variable(gts)
        if train_args['use_gpu']:
            inputs = Variable(inputs).to(device)
            gts = Variable(gts).to(device).long()

        outputs = model(inputs)
        predictions = outputs.data[:].max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()

        val_loss.update(criterion(outputs, gts).data/N, N)
        # val_loss.update(criterion(outputs, gts).data, N)
        if random.random() > train_args['val_img_sample_rate']:
            inputs_all.append(None)
        else:
            inputs_all.append(inputs.data[:].squeeze_(0).cpu())
        gts_all.append(gts.data[:].squeeze_(0).cpu().numpy())
        predictions_all.append(predictions)

    acc, acc_cls, mean_iu, fwavacc,single_cls_acc,kappa = evaluate(predictions_all, gts_all, train_args['num_class'])

    if mean_iu > train_args['best_record']['mean_iu']:
        train_args['best_record']['val_loss'] = val_loss.avg
        train_args['best_record']['epoch'] = epoch
        train_args['best_record']['acc'] = acc
        train_args['best_record']['acc_cls'] = acc_cls
        train_args['best_record']['mean_iu'] = mean_iu
        train_args['best_record']['fwavacc'] = fwavacc
        #
    # print(optimizer.param_groups)
    snapshot_name = 'epoch_%d_loss_%.5f_acc_%.5f_acc-cls_%.5f_mean-iu_%.5f_fwavacc_%.5f_lr_%.7f_kappa_%.5f' % (
            epoch, val_loss.avg, acc, acc_cls, mean_iu, fwavacc,optimizer.param_groups[0]['lr'],kappa)
    """###############-------save models and pred image--------###############"""
    if epoch % 30==0:
        if train_args['val_save_to_img_file']:
            to_save_dir = os.path.join(train_args['ckpt_path'], args['exp'],
                                       train_args['exp_name'] + 'pre_img_epoch %d' % epoch)
            check_mkdir(to_save_dir)


        torch.save(model.state_dict(),
                   os.path.join(train_args['ckpt_path'], args['exp'], train_args['exp_name'], snapshot_name + '.pth'))
        torch.save(optimizer.state_dict(),
                   os.path.join(train_args['ckpt_path'], args['exp'], train_args['exp_name'],
                                'opt_' + snapshot_name + '.pth'))
        val_visual = []
        for idx, data in enumerate(zip(inputs_all, gts_all, predictions_all)):
            if data[0] is None:
                continue
            input_pil = restore(data[0])##反归一化成原图片
            gt_pil = dataset_2_cls_4channel.colorize_mask(data[1])##上色
            predictions_pil = dataset_2_cls_4channel.colorize_mask(data[2])

            if train_args['val_save_to_img_file']:

                input_pil.save(os.path.join(to_save_dir, '%d_input.png' % idx))
                predictions_pil.save(os.path.join(to_save_dir, '%d_prediction.png' % idx))
                gt_pil.save(os.path.join(to_save_dir, '%d_gt.png' % idx))
            val_visual.extend([visualize(input_pil.convert('RGB')), visualize(gt_pil.convert('RGB')),
                            visualize(predictions_pil.convert('RGB'))])

        val_visual = torch.stack(val_visual, 0)
        val_visual = vutils.make_grid(val_visual, nrow=3, padding=5)
        writer.add_image(snapshot_name, val_visual)

    print('--------------------------------starting validating-----------------------------------')
    print('val_single_cls_acc:')
    print(single_cls_acc)
    print('[epoch %d], [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f],[kappa %.5f]' % (
        epoch, val_loss.avg, acc, acc_cls, mean_iu, fwavacc,kappa))

    print('best record: [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f], [epoch %d]' % (
        train_args['best_record']['val_loss'], train_args['best_record']['acc'], train_args['best_record']['acc_cls'],
        train_args['best_record']['mean_iu'], train_args['best_record']['fwavacc'], train_args['best_record']['epoch']))

    print('--------------------------------finish validating-------------------------------------')
    writer.add_scalars('single_cls_acc', {'background': single_cls_acc[0], train_args['training_cls']: single_cls_acc[1]}, epoch)
    writer.add_scalar('val_loss', val_loss.avg, epoch)
    writer.add_scalar('acc', acc, epoch)
    writer.add_scalar('acc_cls', acc_cls, epoch)
    writer.add_scalar('mean_iu', mean_iu, epoch)
    writer.add_scalar('fwavacc', fwavacc, epoch)
    writer.add_scalar('lr',optimizer.param_groups[0]['lr'],epoch)
    writer.add_scalar('kappa',kappa,epoch)

    model.train()
    return val_loss.avg



def adjust_lr(optimizer, epoch):
    ##set new lr fo every 30 epoch
    lr = args['lr'] * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':

    for a in ['field','building']:  ##['building','grass','waterbody','field','road','forest','bareland']
        args['training_cls']=a
        args['exp']='exp_%s_deeplabv3_4channel' % a
        args['exp_name']='train_deeplabv3_4channel_%s' %a
        writer = SummaryWriter(os.path.join(args['ckpt_path'], args['exp'], args['exp_name']))
        main(args)

