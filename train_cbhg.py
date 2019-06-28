#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 23:14:41 2019

@author: xpwang
"""
from data import HanziDataset, PYHZDataLoader
from network import MyNet
from hyperparams import Hyperparams as hp
import torch
from torch.autograd import Variable
from torch import optim
import torch.nn as nn
import os

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
if __name__=='__main__':
    #build_corpus()
    train_dataset = HanziDataset(tsv_file='data/zh.tsv')
    train_loader  = PYHZDataLoader(train_dataset, 
                                   batch_size=8,
                                   num_workers=4)
    model = MyNet(len(train_dataset.pnyn2idx), hp.embed_size, len(train_dataset.hanzi2idx))
    criterion = nn.CrossEntropyLoss()
    # Make optimizer
    optimizer = optim.Adam(model.parameters(), lr=hp.lr)
    
    print(model)
    print("Number of parameters: %d" % MyNet.get_param_size(model))
    
    model = model.cuda()
    
    # Load checkpoint if exists
    try:
        checkpoint = torch.load('first_save.pth.tar')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("\n--------model restored --------\n")

    except:
        print("\n--------Start New Training--------\n")
    
    criterion = criterion.cuda()
    
    lossv = AverageMeter()
    #=========================model training==============================#
    model.train()
    for epoch in range(hp.num_epochs):
        cnt = 0
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            #print(data[0], data[1])
            characters = Variable(data[0].type(torch.LongTensor), requires_grad=False).cuda()
            targets    = Variable(data[1].type(torch.LongTensor), requires_grad=False).cuda()
            
            output = model.forward(characters)
            losses = 0
            for jj in range(output.size()[0]):
                losses += criterion(output[jj], targets[jj])
            losses /= output.size()[0]
            
            cnt += output.size()[0]
            loss_value = losses.item()
            lossv.update(loss_value, output.size()[0])
            #if i%100==0:                
            losses.backward()
            nn.utils.clip_grad_norm(model.parameters(), 1.)
            optimizer.step()
            if (i+1)%50 == 0:
                #print('loss = ', losses.item())
                rec_file = open('Train_Report_LM.txt', 'a', encoding='UTF-8')
                txt = 'epoch = %d  # of sample = %d' %(epoch+1, cnt)
                txt += '  Loss = %f' %(lossv.val)
                txt += '  Loss_ave = %f\n' %(lossv.avg)
                rec_file.write(txt)
                print(txt)
                rec_file.close()
            if (i+1)%500 == 0:
                save_checkpoint({'model':model.state_dict(),
                                 'optimizer':optimizer.state_dict()},
        os.path.join(hp.checkpoint_path,'checkpoint_epoch%d_iter%d.pth.tar' % (epoch, i+1)))
    #=========================model testing=============================#
    model.eval()
    for i, data in enumerate(train_loader):
        if i==200:
            break
        #print(data[0], data[1])
        characters = Variable(data[0].type(torch.LongTensor), requires_grad=False).cuda()
        targets    = Variable(data[1].type(torch.LongTensor), requires_grad=False).cuda()
        output = model(characters)
        dec = torch.argmax(output.cpu(), dim=2)
        targets = targets.cpu()
        for j in range(output.size()[0]):
            pre_hz = []
            ref_hz = []
            for idx in range(len(dec[j])):
                pre_hz.append(train_dataset.idx2hanzi[dec[j][idx].item()])
                ref_hz.append(train_dataset.idx2hanzi[targets[j][idx].item()])
        pre = ''.join(pre_hz)
        ref = ''.join(ref_hz)
        print(pre, ref)
        #print(output[0], targets[0])