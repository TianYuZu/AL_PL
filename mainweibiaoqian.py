'''Active Learning Procedure in PyTorch.

Reference:
[Yoo et al. 2019] Learning Loss for Active Learning (https://arxiv.org/abs/1905.03677)
'''

# Python
import os
import random

# Torch
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
from util import *
# Torchvison
import torchvision.transforms as T
import torchvision.models as models
from torchvision.datasets import CIFAR100

# Utils
# import visdom
from tqdm import tqdm

# Custom
import models.resnetPEDCC as resnet
from config import *
from data.sampler import SubsetSequentialSampler

from util import  read_pkl
from datetime import datetime
import pickle
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,6,7'

# Seed
random.seed("Inyoung Cho")
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True


##
# Data
train_transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomCrop(size=32, padding=4),
    T.ToTensor(),
    T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100
])

test_transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100
])

cifar100_train = CIFAR100('../cifar100', train=True, download=True, transform=train_transform)
cifar100_unlabeled   = CIFAR100('../cifar100', train=True, download=True, transform=test_transform)
cifar100_test  = CIFAR100('../cifar100', train=False, download=True, transform=test_transform)


##
# Loss Prediction Loss
def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape
    
    input = (input - input.flip(0))[:len(input)//2] # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target)//2]
    target = target.detach()

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1 # 1 operation which is defined by the authors
    
    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0) # Note that the size of input is already halved
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()
    
    return loss


##
# Train Utils
iters = 0


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)

    normp = torch.sum(buffer, 1).add_(1e-10)
    norm = torch.sqrt(normp)

    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output


def train_epoch(model, loss_fn,loss_fn2, optimizer, dataloaders, epoch, epoch_loss):

    model.train()
    global iters
    for data in tqdm(dataloaders['train'], leave=False, total=len(dataloaders['train'])):
        inputs = data[0].cuda()
        labels = data[1].cuda()
        iters += 1

        # read PEDCC weights
        map_dict = read_pkl()
        tensor_empty = torch.Tensor([]).cuda()
        for target_index in labels:
            tensor_empty = torch.cat((tensor_empty, torch.from_numpy(map_dict[target_index.item()]).float().cuda()), 0)
        label_mse_tensor = tensor_empty.view(-1, 512)
        label_mse_tensor = label_mse_tensor.cuda()  # PEDCC of each class

        # forward
        pred,fea,fea_norm = model(inputs)
        t_loss1 = loss_fn(pred, labels)  # PEDCC-AMSOFTMAX
        label_mse_tensor=l2_norm(label_mse_tensor)
        t_loss2 = loss_fn2(fea_norm, label_mse_tensor)
        t_loss2 =10*torch.pow(t_loss2,0.5)
        loss = t_loss1 + t_loss2

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


#
def test(model, dataloaders, mode='val'):
    assert mode == 'val' or mode == 'test'
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for (inputs, labels) in dataloaders[mode]:
            inputs = inputs.cuda()
            labels = labels.cuda()

            scores, _,_ = model(inputs)
            _, preds = torch.max(scores.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    
    return 100 * correct / total

#
def train(model, loss_fn,loss_fn2, optimizer, scheduler, dataloaders, num_epochs, epoch_loss):
    print('>> Train a Model.')
    f.write('>> Train a Model.')
    best_acc = 0.
    checkpoint_dir = os.path.join('./cifar100', 'train', 'weights')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    for epoch in range(num_epochs):
        scheduler.step()
        train_epoch(model, loss_fn,loss_fn2, optimizer, dataloaders, epoch, epoch_loss)
        #print('测试是否运行1')
        # Save a checkpoint
        if False and epoch % 5 == 4:
            #print('测试是否运行2')
            acc = test(model, dataloaders, 'test')
            if best_acc < acc:
                best_acc = acc
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict_backbone': model.state_dict(),

                },
                '%s/active_resnet18_cifar100.pth' % (checkpoint_dir))
            print('Val Acc: {:.3f} \t Best Acc: {:.3f}'.format(acc, best_acc))
    print('>> Finished.')
    f.write('>> Finished.')

def cos_dis(a, b):
    ##求模
    buffer = torch.pow(a, 2)
    a_norm = torch.sqrt(torch.sum(buffer).add_(1e-10))

    b_buffer = torch.pow(b, 2)
    b_norm = torch.sqrt(torch.sum(b_buffer).add_(1e-10))

    ##向量相乘
    vector_mul = a.mul(b)
    vector_mul = torch.sum(vector_mul)

    ##投影
    projection = torch.div(vector_mul, a_norm)
    projection = torch.div(projection, b_norm)
    return projection


def get_max_cosdis(output, features):
    total = output.shape[0]
    maxDisList = []
    for i in range(total):
        dis=torch.tensor([]).cuda()
        dis_list=[]
        for ui_label in features.values():
            ##归一化ui_label[0]
            feature=ui_label[0].float().cuda()
            # print(torch.norm(feature))
            feature=torch.div(feature,torch.norm(feature))
            #print(torch.norm(feature))
            distance =cos_dis(output[i],feature)
            dis_list.append(distance.item())

        maxDisList.append(max(dis_list))
    return maxDisList


def get_margin_cosdis(output, features):
    total = output.shape[0]
    marginDisList = []
    for i in range(total):
        dis=torch.tensor([]).cuda()
        dis_list=[]
        for ui_label in features.values():
            ##归一化ui_label[0]
            feature=ui_label[0].float().cuda()
            # print(torch.norm(feature))
            feature=torch.div(feature,torch.norm(feature))
            #print(torch.norm(feature))
            distance =cos_dis(output[i],feature)
            dis_list.append(distance.item())

        marginDisList.append(margin_cosdis(dis_list))
    return marginDisList

def get_norm_cosdis(output, features):
    total = output.shape[0]
    normDisList = []
    for i in range(total):
        dis=torch.tensor([]).cuda()
        dis_list=[]
        for ui_label in features.values():
            ##归一化ui_label[0]
            feature=ui_label[0].float().cuda()
            # print(torch.norm(feature))
            feature=torch.div(feature,torch.norm(feature))
            #print(torch.norm(feature))
            distance =cos_dis(output[i],feature)
            dis_list.append(distance.item())

        # tmpNorm=torch.norm(torch.Tensor(dis_list))
        tmpNorm=torch.sum(torch.pow(torch.Tensor(dis_list),2))
        normDisList.append(tmpNorm.item())
    return normDisList


def margin_cosdis(list):
    Fmax=-100
    Smax=-100
    for item in list:
        if(item>Fmax):
            Smax = Fmax
            Fmax=item
        elif(Smax<item<=Fmax):
            Smax = item
    return Fmax-Smax

#
def get_uncertainty(model, unlabeled_loader):
    model.eval()
    uncertainty1 = torch.tensor([]).cuda()
    uncertainty2 = torch.tensor([]).cuda()
    uncertainty3 = torch.tensor([]).cuda()

    with torch.no_grad():
        for (inputs, labels) in unlabeled_loader:
            inputs = inputs.cuda()
            # labels = labels.cuda()

            pred, fea, fea_norm = model(inputs)

            # ##归一化前的特征的模
            # fea_pow=torch.pow(fea,2)
            # fea_mo=torch.sqrt(torch.sum(fea_pow,dim=1).add(1e-10))
            # metric1=fea_mo
            # # metric1_max=torch.max(metric1)
            # # metric1_min = torch.min(metric1)
            #
            # ###归一化后特征的余弦距离最大值
            # map_dict = read_pkl()
            # cosDis=get_max_cosdis(fea_norm,map_dict)
            # metric2=torch.tensor(cosDis).cuda()  ## 取最大值 size B
            # # metric2_max=torch.max(metric2)
            # # metric2_min = torch.min(metric2)
            #
            #
            # #metric=metric1*metric2

            # ###归一化后特征的余弦距离最大值与次大值的差
            # map_dict = read_pkl()
            # NormcosDis=get_norm_cosdis(fea_norm,map_dict)
            # metric3=torch.tensor(NormcosDis).cuda()  ## 取最大值 size B

            # uncertainty1 = torch.cat((uncertainty1, metric1), 0)
            # uncertainty2 = torch.cat((uncertainty2, metric2), 0)

            metric1=torch.max(pred,dim=1)[0]

            uncertainty3 = torch.cat((uncertainty3, metric1), 0)
    return uncertainty3.cpu()


def get_pseudo_label(pseudo_dataloader):
    


    pseudo_labels=torch.LongTensor([]).cuda()

    pseudo_labels=[]
    model.eval()
    with torch.no_grad():
        for (inputs, labels) in pseudo_dataloader:
            inputs = inputs.cuda()


            scores, _,_ = model(inputs)
            _, preds = torch.max(scores.data, 1)

            # 1pseudo_labels = torch.cat((pseudo_labels, preds), 0)

            pseudo_labels=pseudo_labels+preds.cpu().numpy().tolist()

    return pseudo_labels



# Main
if __name__ == '__main__':

    f=open(os.path.join('./log/newpedcc','logweibiaoqian2.txt'),'w')
    startTime=datetime.now()


    for trial in range(TRIALS):
        # Initialize a labeled dataset by randomly sampling K=ADDENDUM=1,000 data points from the entire dataset.
        indices = list(range(NUM_TRAIN))
        random.shuffle(indices)
        labeled_set = indices[:ADDENDUM_start]
        unlabeled_set = indices[ADDENDUM_start:]
        
        train_loader = DataLoader(cifar100_train, batch_size=BATCH,
                                  sampler=SubsetRandomSampler(labeled_set), 
                                  pin_memory=True)
        test_loader  = DataLoader(cifar100_test, batch_size=BATCH)
        dataloaders  = {'train': train_loader, 'test': test_loader}
        
        # Model
        model= resnet.ResNet18(num_classes=100)
        # total_num=sum(p.numel() for p in model.parameters)
        # trainable_num=sum(p.numel() for p in model.parameters if p.requires_grad)
        # print("total_num"+total_num)
        # print("trainable_num" + trainable_num)

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
        model=model.cuda()
        torch.backends.cudnn.benchmark = False

        # Active learning cycles
        for cycle in range(CYCLES):
            # Loss, criterion and scheduler (re)initialization
            loss_fn = get_loss_fn()
            loss_fn2 = torch.nn.MSELoss()

            optimizer = optim.SGD(model.parameters(), lr=LR,
                                    momentum=MOMENTUM, weight_decay=WDECAY)
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES)  ##学习率设置

            # Training and test
            train(model, loss_fn, loss_fn2,optimizer, scheduler, dataloaders, EPOCH, EPOCHL)
            acc = test(model, dataloaders, mode='test')
            log_str=('Trial {}/{} || Cycle {}/{} || Label set size {}: Test acc {}'.format(trial+1, TRIALS, cycle+1, CYCLES, len(labeled_set), acc))
            print(log_str)
            f.write(log_str+'\n')

            ##
            #  Update the labeled dataset via loss prediction-based uncertainty measurement

            # Randomly sample 10000 unlabeled data points
            random.shuffle(unlabeled_set)
            subset = unlabeled_set[:SUBSET]

            # # Create unlabeled dataloader for the unlabeled subset
            unlabeled_loader = DataLoader(cifar100_unlabeled, batch_size=BATCH,
                                          sampler=SubsetSequentialSampler(subset), # more convenient if we maintain the order of subset
                                          pin_memory=True)

            # Measure uncertainty of each data points in the subset
            uncertainty3 = get_uncertainty(model, unlabeled_loader)
            arg = np.argsort(uncertainty3)

            #
            #
            # arg1 = np.argsort(uncertainty1)
            # arg2 = np.argsort(uncertainty2)
            #
            #根据两次排名再排名
            # dic = {}
            # for i in range(SUBSET):
            #     idx1 = i
            #     idx2 = np.where(arg2 == arg1[i])[0][0]
            #     dic[arg1[i]] = idx1 + idx2
            #
            # res = sorted(dic.items(), key=lambda kv: (kv[1], kv[0]))
            # arg = [x[0] for x in res]

            #随机挑选
            # arg=[i for i in range (SUBSET)]
            # random.shuffle(arg)

            
            # Update the labeled dataset and the unlabeled dataset, respectively
            # labeled_set += list(torch.tensor(subset)[arg][:ADDENDUM].numpy())
            # unlabeled_set = list(torch.tensor(subset)[arg][ADDENDUM:].numpy()) + unlabeled_set[SUBSET:]

            labeled_set += list(torch.tensor(subset)[arg][:ADDENDUM].numpy())
            pseudo_set = list(torch.tensor(subset)[arg][-Pseudo_num:].numpy())
            unlabeled_set = list(torch.tensor(subset)[arg][ADDENDUM:].numpy()) + unlabeled_set[SUBSET:]

            # Create a new dataloader for the updated labeled dataset

            pseudo_loader = DataLoader(cifar100_unlabeled, batch_size=BATCH,
                                       sampler=SubsetSequentialSampler(pseudo_set),
                                       # more convenient if we maintain the order of subset
                                       pin_memory=True)
            pseudo_labels = get_pseudo_label(pseudo_loader)
            labeled_set += pseudo_set

            cifar100_train_copy = cifar100_train

            for idx in range(len(pseudo_set)):
                dataset1 = cifar100_train_copy.targets
                dataset2 = cifar100_train.targets

                cifar100_train_copy.targets[pseudo_set[idx]] = pseudo_labels[idx]

            # Create a new dataloader for the updated labeled dataset
            dataloaders['train'] = DataLoader(cifar100_train, batch_size=BATCH,
                                              sampler=SubsetRandomSampler(labeled_set), 
                                              pin_memory=True)
        
        # Save a checkpoint
        torch.save({
                    'trial': trial + 1,
                    'state_dict_backbone': model.state_dict(),
                },
                './cifar100/train/weights/active_resnet18_cifar100_trial{}.pth'.format(trial))
    endTime = datetime.now()
    time=(endTime-startTime).seconds
    print("time: "+str(time)+"s")
    f.write("time: "+str(time)+"s" + '\n')
    f.close()