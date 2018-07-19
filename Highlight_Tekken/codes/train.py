import os, time, glob
from itertools import chain

import numpy as np

import itertools, time, os
import torch
from torch import nn
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.optim as optim
import random
import torch.backends.cudnn as cudnn

from models.bidirectional import C3D, GRU
from config import get_config
from videoloader import get_loader

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
            for step, (h, r) in enumerate(zip(self.h_loader, self.r_loader)):
                h_video = h
                r_video = r

                # for highlight video
                h_video = Variable(h_video.cuda())
                self.gru.zero_grad()

                # forward
                predicted = self.gru(h_video.cuda()) # predicted snippet's score
                target = torch.from_numpy(np.ones([len(predicted)], dtype=np.float)).cuda() # highlight videos => target:1
                h_loss = Variable(criterion(predicted, target), requires_grad=True) # compute loss

                h_loss.backward()
                opt.step()

                # for raw video
                r_video = Variable(r_video.cuda())
                self.gru.zero_grad()

                # forward
                predicted = self.gru(r_video.cuda())  # predicted snippet's score
                target = torch.from_numpy(np.zeros([len(predicted)], dtype=np.float)).cuda() # row videos => target:0
                r_loss = Variable(criterion(predicted, target), requires_grad=True) # compute loss

                r_loss.backward()
                opt.step()

                step_end_time = time.time()

                # print logging
                print('[%d/%d][%d/%d] - time: %.2f, h_loss: %.3f, r_loss: %.3f, total_loss: %.3f'
                      % (epoch + 1, self.n_epochs, step + 1, min(len(self.h_loader), len(self.r_loader)),
                         step_end_time - start_t, h_loss, r_loss, h_loss + r_loss))

                # validating for test dataset
                # compute predicted score accuracy
                if step % self.log_interval == 0:
                    # for step, t in enumerate(self.test_loader):
                    #     t_video = t[0]
                    #     t_label = t[1]


                    pass

            if epoch % self.checkpoint_step == 0:
                torch.save(self.gru.state_dict(), '../checkpoints/epoch' + str(epoch) + '.pth')
                print("checkpoint saved!")

"""
한 Epoch 돌려본 결과 =>
[1/10][1/25] - time: 20.92, h_loss: -0.000, r_loss: -0.000, total_loss: -0.000
[1/10][2/25] - time: 32.44, h_loss: 5.526, r_loss: 27.631, total_loss: 33.157
[1/10][3/25] - time: 40.58, h_loss: -0.000, r_loss: 27.631, total_loss: 27.631
[1/10][4/25] - time: 48.98, h_loss: -0.000, r_loss: 27.631, total_loss: 27.631
[1/10][5/25] - time: 63.64, h_loss: -0.000, r_loss: 27.631, total_loss: 27.631
[1/10][6/25] - time: 72.37, h_loss: -0.000, r_loss: 27.631, total_loss: 27.631
[1/10][7/25] - time: 84.23, h_loss: 5.024, r_loss: 27.631, total_loss: 32.655
[1/10][8/25] - time: 94.42, h_loss: -0.000, r_loss: 27.631, total_loss: 27.631
[1/10][9/25] - time: 106.58, h_loss: -0.000, r_loss: 27.631, total_loss: 27.631
[1/10][10/25] - time: 118.88, h_loss: -0.000, r_loss: 27.631, total_loss: 27.631
[1/10][11/25] - time: 128.95, h_loss: -0.000, r_loss: 27.631, total_loss: 27.631
[1/10][12/25] - time: 137.09, h_loss: -0.000, r_loss: -0.000, total_loss: -0.000
[1/10][13/25] - time: 150.08, h_loss: -0.000, r_loss: 27.631, total_loss: 27.631
[1/10][14/25] - time: 160.26, h_loss: -0.000, r_loss: 27.631, total_loss: 27.631
[1/10][15/25] - time: 178.51, h_loss: -0.000, r_loss: 27.631, total_loss: 27.631
[1/10][16/25] - time: 189.98, h_loss: -0.000, r_loss: 27.631, total_loss: 27.631
[1/10][17/25] - time: 198.04, h_loss: -0.000, r_loss: 27.631, total_loss: 27.631
[1/10][18/25] - time: 213.45, h_loss: -0.000, r_loss: 20.723, total_loss: 20.723
[1/10][19/25] - time: 225.20, h_loss: -0.000, r_loss: 27.631, total_loss: 27.631
[1/10][20/25] - time: 238.89, h_loss: -0.000, r_loss: 27.631, total_loss: 27.631
[1/10][21/25] - time: 249.47, h_loss: -0.000, r_loss: 27.631, total_loss: 27.631
[1/10][22/25] - time: 264.77, h_loss: -0.000, r_loss: 27.631, total_loss: 27.631
[1/10][23/25] - time: 278.96, h_loss: -0.000, r_loss: 27.631, total_loss: 27.631
[1/10][24/25] - time: 290.43, h_loss: -0.000, r_loss: 27.631, total_loss: 27.631
[1/10][25/25] - time: 306.33, h_loss: -0.000, r_loss: 27.631, total_loss: 27.631

==> model.py에서 forwarding/ backwarding 한번이라도 snippet score이 1로 측정되면 전부 highlight라고 간주하게 했는데
그래서 row video에게 너무 각박한가...? 어려워요,,
"""