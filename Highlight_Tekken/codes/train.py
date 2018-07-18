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
    def __init__(self, config, h_loader, r_loader):
        self.config = config

        self.h_loader = h_loader
        self.r_loader = r_loader

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
        self.gru.train() # model: training mode

        for epoch in range(self.n_epochs):
            # common_len = min(len(self.h_loader),len(self.r_loader))
            for step, (h, r) in enumerate(zip(self.h_loader, self.r_loader)):
                h_video = h
                r_video = r

                # for highlight video
                h_video = Variable(h_video.cuda())

                self.gru.zero_grad()
                h_loss = Variable(self.gru(h_video, 'HV').cuda(), requires_grad=True) # target: HV
                h_loss.backward()
                opt.step()

                # for raw video
                r_video = Variable(r_video.cuda())

                self.gru.zero_grad()
                r_loss = Variable(self.gru(r_video, 'RV').cuda(), requires_grad=True) # target: RV
                r_loss.backward()
                opt.step()

                step_end_time = time.time()

                # print logging
                print('[%d/%d][%d/%d] - time: %.2f, h_loss: %.3f, r_loss: %.3f'
                      % (epoch + 1, self.n_epochs, step + 1, min(len(self.h_loader), len(self.r_loader)),
                         step_end_time - start_t, h_loss, r_loss))

                if step % self.log_interval == 0: # validating for test dataset
                    # for step, t in enumerate(self.test_loader):
                    #     t_video = t[0]
                    #     t_label = t[1]
                    #
                    #
                    pass

            if epoch % self.checkpoint_step == 0:
                self.gru.save_state_dict('../checkpoints/ckpt' + str(epoch) + '.pt') # save model checkpoint

if __name__ == "__main__":
    config = get_config()

    if config.outf is None:
        config.outf = 'samples'
    os.system('mkdir {0}'.format(config.outf))

    config.manual_seed = random.randint(1, 10000)
    print("Random Seed: ", config.manual_seed)
    random.seed(config.manual_seed)
    torch.manual_seed(config.manual_seed)

    if config.cuda:
        torch.cuda.manual_seed_all(config.manual_seed)

    cudnn.benchmark = True

    dataroot = config.dataroot
    h_datapath = os.path.join(dataroot, "HV")
    r_datapath = os.path.join(dataroot, "RV")
    #t_datapath = os.path.join(dataroot, 'testRV')

    # dataroot, cache, image_size, n_channels, image_batch, video_batch, video_length):
    h_loader, r_loader = get_loader(h_datapath, r_datapath)

    train = Trainer(config, h_loader, r_loader)
    train.train()

