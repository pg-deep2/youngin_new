import torch.nn as nn
import torch

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

h_video_path = '../../dataset/HV'
r_video_path = '../../dataset/RV'
weight_path = '../../pretrained-weight/c3d.pickle'

class GRU(nn.Module):
    def __init__(self, c3d):
        super(GRU, self).__init__()
        self.c3d = c3d
        self.gru = nn.GRU(243, 2, bidirectional=True)
        self.temporal_pool = nn.MaxPool1d(4, 4, 0)

    def forward(self, input):
        start = 0
        end = 48
        e_t = torch.FloatTensor(128, 2).normal_().cuda()

        step = 0
        while end < input.shape[2]:
            x = input[:, :, start:end, :, :]
            # x.shape: 1, 3, 96, h, w
            h = self.c3d(x)
            # h.shape: 1, 512, 3, 9, 9
            h = h.squeeze()
            h = h.view(1, 512, -1).permute(0, 2, 1)
            h = self.temporal_pool(h).permute(0, 2, 1).squeeze()
            # print("h",h.shape)
            # h.shape: 128, 243

            out = (self.gru(h.cuda(), e_t))
            # print("et",e_t.shape)
            # e_t.shape: 128,20

            start += 6
            end += 6
            step += 1

        return out