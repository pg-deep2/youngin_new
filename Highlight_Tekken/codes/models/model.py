######### non-bidirectional version code #########

import torch.nn as nn
import torch

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from videoloader import get_loader

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

h_video_path = '../../dataset/HV'
r_video_path = '../../dataset/RV'
weight_path = '../../weight/c3d.pickle'

##############################################################
######  source: https://github.com/DavideA/c3d-pytorch  ######
##############################################################
class C3D(nn.Module):
    """
    The C3D network as described in [1].
    """

    def __init__(self):
        super(C3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 487)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

# GRU Cell로 구현 시, Bidirectional 간단하게 True/False로 선택 못함
# 따로 구현하는 방법? forward로 한번 통과 backward로 한번 통과?
class GRU(nn.Module):
    def __init__(self, c3d):
        super(GRU, self).__init__()

        self.c3d = c3d
        self.gru = nn.GRUCell(243, 1).cuda()

    def forward(self, input):
        start = 0
        end = 48
        h_t = torch.FloatTensor(128, 1).normal_().cuda() # (batch, hidden)

        input = input.permute(0, 2, 1, 3, 4) # [batch, channel, depth, height, width]
        temporal_pool = nn.MaxPool1d(4, 4, 0)

        step = 0
        while end < input.shape[2]:
            x = input[:, :, start:end, :, :] # x.shape: 1, 3, 48, h, w
            h = self.c3d(x) # c3d forwarding => 1, 512, 3, 9, 9
            h = h.squeeze()
            h = h.view(1, 512, -1).permute(0, 2, 1)
            h = temporal_pool(h).permute(0, 2, 1).squeeze()

            h_t = (self.gru(h.cuda(), h_t))
            print("snippet:", step, h_t) # 매 snippet마다 같은 h_t가 나오는 것 같은데 그래도 되는건가??

            start += 6
            end += 6
            step += 1

        print(len(h_t)) # 128
        return h_t

if __name__ == '__main__':
    # load videos
    h_loader, r_loader = get_loader(h_video_path, r_video_path)

    for idx, (frames, scores) in enumerate(r_loader):
        frames = frames.cuda()
        scores = scores.cuda()
        break

    # define C3D layer
    c3d = C3D()
    c3d.load_state_dict(torch.load(weight_path))  # load pre-trained weight
    c3d = c3d.cuda()

    # remove c3d fc layers
    fc_removed = list(c3d.children())[:-6]

    _p3d_net = []
    relu = nn.ReLU()

    for layer in fc_removed:
        for param in layer.parameters():
            param.requires_grad = False  # no training
        if layer.__class__.__name__ == 'MaxPool3d':
            _p3d_net.extend([layer, relu])  # add activation function
        else:
            _p3d_net.append(layer)

    c3d = nn.Sequential(*_p3d_net).cuda()  # new p3d net

    gru = GRU(c3d)
    out = gru(frames)

    print(out) # tuple: length 128 for one video