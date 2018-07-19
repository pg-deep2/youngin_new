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

        # define loss function
        criterion = nn.BCELoss()
        self.gru.train() # model: training mode

        for epoch in range(self.n_epochs):
            # common_len = min(len(self.h_loader),len(self.r_loader))
            for step, (h, r) in enumerate(zip(self.h_loader, self.r_loader)):
                h_video = h
                r_video = r

                # for highlight video
                h_video = Variable(h_video.cuda())

                self.gru.zero_grad()
                #opt.zero_grad()

                # forward
                predicted = self.gru(h_video.cuda()) # predicted snippet's score
                target = torch.from_numpy(np.ones([len(predicted)], dtype=np.float)).cuda() # highlight videos => target:1
                h_loss = Variable(criterion(predicted, target), requires_grad=True) # compute loss

                h_loss.backward()
                opt.step()

                # for raw video
                r_video = Variable(r_video.cuda())

                self.gru.zero_grad()
                #opt.zero_grad()

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
    t_datapath = os.path.join(dataroot, 'testRV')

    # dataroot, cache, image_size, n_channels, image_batch, video_batch, video_length):
    h_loader, r_loader, test_loader = get_loader(h_datapath, r_datapath, t_datapath)

    train = Trainer(config, h_loader, r_loader, test_loader)
    train.train()


    """
    [1/10][1/26] - time: 14.71, h_loss: -0.000, r_loss: 27.631, total_loss: 27.631
[1/10][2/26] - time: 28.15, h_loss: -0.000, r_loss: 24.868, total_loss: 24.868
[1/10][3/26] - time: 40.11, h_loss: 27.631, r_loss: 27.631, total_loss: 55.262
[1/10][4/26] - time: 48.74, h_loss: -0.000, r_loss: 27.631, total_loss: 27.631
[1/10][5/26] - time: 55.60, h_loss: 20.723, r_loss: 20.723, total_loss: 41.447
[1/10][6/26] - time: 64.12, h_loss: 9.210, r_loss: 27.631, total_loss: 36.841
[1/10][7/26] - time: 74.86, h_loss: 27.631, r_loss: 27.631, total_loss: 55.262
[1/10][8/26] - time: 86.69, h_loss: 24.561, r_loss: 27.631, total_loss: 52.192
[1/10][9/26] - time: 99.67, h_loss: -0.000, r_loss: 27.631, total_loss: 27.631
[1/10][10/26] - time: 110.49, h_loss: 13.816, r_loss: 27.631, total_loss: 41.447
[1/10][11/26] - time: 123.14, h_loss: 1.727, r_loss: 27.631, total_loss: 29.358
[1/10][12/26] - time: 137.44, h_loss: 10.048, r_loss: 8.289, total_loss: 18.337
[1/10][13/26] - time: 151.24, h_loss: 5.526, r_loss: 27.631, total_loss: 33.157
[1/10][14/26] - time: 161.74, h_loss: 19.342, r_loss: -0.000, total_loss: 19.342
[1/10][15/26] - time: 174.55, h_loss: -0.000, r_loss: 27.631, total_loss: 27.631
[1/10][16/26] - time: 186.88, h_loss: -0.000, r_loss: 10.362, total_loss: 10.362
[1/10][17/26] - time: 202.98, h_loss: 27.631, r_loss: 27.631, total_loss: 55.262
[1/10][18/26] - time: 211.62, h_loss: -0.000, r_loss: 2.763, total_loss: 2.763
    
    """

