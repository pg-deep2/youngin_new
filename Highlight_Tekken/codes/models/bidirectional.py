######### Bidirectional version code #########
import torch.nn as nn
import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

# Bidirectional GRU
# forwarding once, backwarding once
class GRU(nn.Module):
    def __init__(self, c3d):
        super(GRU, self).__init__()
        self.c3d = c3d
        self.temporal_pool = nn.MaxPool1d(4, 4, 0).cuda()
        self.gru = nn.GRUCell(243, 10).cuda()
        self.fc1 = nn.Linear(128*10, 10).cuda()
        self.fc2 = nn.Linear(10, 1).cuda()
        self.sigmoid = nn.Sigmoid().cuda()

    def forward(self, input):
        input_cp = input.clone() # copy input tensor for backwarding GRU

        start = 0
        end = 48

        f_ht = torch.FloatTensor(128, 10).normal_().cuda() # (batch, hidden)
        b_ht = torch.FloatTensor(128, 10).normal_().cuda() # (batch, hidden)

        f_snp_score_list = []
        b_snp_score_list = []

        # forwarding once
        input = input.permute(0, 2, 1, 3, 4)  # [batch, channel, depth, height, width]
        forward_step = 0

        while end < input.shape[2]:
            x = input[:, :, start:end, :, :] # x.shape: 1, 3, 48, h, w
            h = self.c3d(x) # c3d forwarding => 1, 512, 3, 9, 9
            h = h.squeeze()
            h = h.view(1, 512, -1).permute(0, 2, 1)
            h = self.temporal_pool(h).permute(0, 2, 1).squeeze()

            f_ht = (self.gru(h.cuda(), f_ht)) # [128, 10]
            f_ht_flatten = f_ht.view(1, -1) # [1, 128*10] # flatten to
            fc1_out = self.fc1(f_ht_flatten)
            fc2_out = self.fc2(fc1_out)
            sigmoid_out = self.sigmoid(fc2_out)

            f_snp_score_list.append(sigmoid_out.item())
            start += 6
            end += 6
            forward_step += 1

        ##########################################################################

        # backwarding once
        input_cp = input_cp.cpu()
        reversed_input = torch.from_numpy(np.flip(input_cp.numpy(), axis=1).copy()) # reverse input depth(axis=1) sequence
        reversed_input = reversed_input.permute(0, 2, 1, 3, 4) # [batch, channel, depth(reversed), h, w]
        reversed_input = reversed_input.cuda()

        start = 0
        end = 48
        backward_step = 0

        while end < reversed_input.shape[2]:
            x = reversed_input[:, :, start:end, :, :]  # x.shape: 1, 3, 48, h, w
            h = self.c3d(x)  # c3d forwarding => 1, 512, 3, 9, 9
            h = h.squeeze()
            h = h.view(1, 512, -1).permute(0, 2, 1)
            h = self.temporal_pool(h).permute(0, 2, 1).squeeze()

            b_ht = (self.gru(h.cuda(), b_ht))
            b_ht_flatten = b_ht.view(1, -1)  # [1, 128*10] # flatten to
            fc1_out = self.fc1(b_ht_flatten)
            fc2_out = self.fc2(fc1_out)
            sigmoid_out = self.sigmoid(fc2_out)


            b_snp_score_list.append(sigmoid_out.item())
            start += 6
            end += 6
            backward_step += 1

        out_score_list = f_snp_score_list + b_snp_score_list
        out_score_list = np.asarray(out_score_list, dtype=np.float)
        out_score_list = torch.from_numpy(out_score_list).cuda()

        return out_score_list