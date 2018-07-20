import os, time, glob
from itertools import chain

import numpy as np
import sys
import itertools, time, os
import torch
from torch import nn
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.optim as optim
import random
import torch.backends.cudnn as cudnn

from models.bidirectional import C3D, GRU
from vis_tool import Visualizer

import time

# -*- coding: cp949 -*-


def denorm(x):
    out = (x+1)/2
    return out.clamp(0, 1)

class Trainer(object):
    def __init__(self, config, h_loader, r_loader, test_loader):
        self.config = config

        self.h_loader = h_loader
        self.r_loader = r_loader
        self.test_loader = test_loader

        # hyper parameters for training
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.weight_decay = config.weight_decay
        self.n_epochs = config.n_epochs

        self.log_interval = int(config.log_interval)
        self.checkpoint_step = int(config.checkpoint_step)

        self.use_cuda = config.cuda
        self.outf = config.outf

        # build c3d -> GRU model
        self.build_model()

        # visualizing tool
        self.vis = Visualizer()

    def build_model(self):
        self.p3d = C3D().cuda() # feature extraction c3d
        self.load_model() # load pretrained weight and remove FC layers

        self.gru = GRU(self.p3d).cuda() # bidirectional GRU

        print("MODEL:")
        print(self.gru)

    def load_model(self):
        self.p3d.load_state_dict(torch.load(self.config.pretrained_path))

        fc_removed = list(self.p3d.children())[:-6] # remove FC layers
        _p3d_net = []
        relu = nn.ReLU().cuda()

        for layer in fc_removed:
            for param in layer.parameters():
                param.requires_grad = False
            if layer.__class__.__name__ == 'MaxPool3d':
                _p3d_net.extend([layer, relu]) # add activation function
            else:
                _p3d_net.append(layer)
        p3d_net = nn.Sequential(*_p3d_net).cuda()

        self.p3d = p3d_net

    def train(self):
        # define optimizer
        opt = optim.Adam(filter(lambda p: p.requires_grad, self.gru.parameters()),
                               lr=self.lr, betas=(self.beta1, self.beta2),
                               weight_decay=self.weight_decay)

        start_t = time.time()

        criterion = nn.BCELoss() # define loss function
        self.gru.train() # model: training mode

        for epoch in range(self.n_epochs):
            # common_len = min(len(self.h_loader),len(self.r_loader))
            try:
                for step, (h, r) in enumerate(zip(self.h_loader, self.r_loader)):
                    h_video = h
                    r_video = r

                    # for highlight video
                    h_video = Variable(h_video.cuda())
                    self.gru.zero_grad()

                    # forward
                    predicted = self.gru(h_video.cuda())  # predicted snippet's score
                    target = torch.from_numpy(
                        np.ones([len(predicted)], dtype=np.float)).cuda()  # highlight videos => target:1
                    h_loss = Variable(criterion(predicted, target), requires_grad=True)  # compute loss

                    h_loss.backward()
                    opt.step()

                    # for raw video
                    r_video = Variable(r_video.cuda())
                    self.gru.zero_grad()

                    # forward
                    predicted = self.gru(r_video.cuda())  # predicted snippet's score
                    target = torch.from_numpy(
                        np.zeros([len(predicted)], dtype=np.float)).cuda()  # row videos => target:0
                    r_loss = Variable(criterion(predicted, target), requires_grad=True)  # compute loss

                    r_loss.backward()
                    opt.step()

                    step_end_time = time.time()

                    total_loss = h_loss + r_loss
                    # print logging
                    print('[%d/%d][%d/%d] - time: %.2f, h_loss: %.3f, r_loss: %.3f, total_loss: %.3f'
                          % (epoch + 1, self.n_epochs, step + 1, min(len(self.h_loader), len(self.r_loader)),
                             step_end_time - start_t, h_loss, r_loss, total_loss))
                    self.vis.plot('Loss with lr=%.4f' % self.lr, (total_loss.data).cpu().numpy())

                    # validating for test dataset
                    # compute predicted score accuracy
                    if step % self.log_interval == 0:
                        # for step, t in enumerate(self.test_loader):
                        #     t_video = t[0]
                        #     t_label = t[1]
                        pass

            # keyboard interrupt routine => save checkpoint file & hyper parameters
            except KeyboardInterrupt:
                print('Keyboard Interrupt!')
                checkpoint_filename = input('Enter checkpoint filaname:')
                torch.save(self.gru.state_dict(),
                            '../checkpoints/exception/' + checkpoint_filename + '.pth')  # save checkpoint

                # save hyperparameters
                hyperparameter_filename = input('Enter hyper parameter filename:')
                with open('../checkpoints/exception/' + hyperparameter_filename + '.txt', 'w') as f:
                    f.writelines("learning rate: %.6f\n" % self.lr)
                    f.writelines("beta1, beta2: %.3f, %.3f\n" % (self.beta1, self.beta2))
                    f.writelines("weight decay rate: %.3f\n" % self.weight_decay)
                    f.writelines("current state: epoch [%d/%d] step [%d/%d]\n" % (
                    epoch + 1, self.n_epochs, step + 1, min(len(self.h_loader), len(self.r_loader))))
                sys.exit(0)

            # save checkpoints
            if epoch % self.checkpoint_step == 0:
                now = time.localtime()
                s = "%04d-%02d-%02d %02d-%02d-%02d" % (
                now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
                torch.save(self.gru.state_dict(), '../checkpoints/epoch' + str(epoch) + '_' + s + '.pth')
                print("checkpoint saved!")
