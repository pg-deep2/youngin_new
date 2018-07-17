import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data

from c3d import C3D
from videoloader import get_loader
from bidir_lstm import GRU
# from single_lstm import single_LSTM

h_video_path = '../../dataset/HV'
r_video_path = '../../dataset/RV'
weight_path = '../../pretrained-weight/c3d.pickle'

# c3d layer + bidirectional-LSTM layer
class HighlightClassifier(nn.Module):
    def __init__(self):
        super(HighlightClassifier, self).__init__()
        # define C3D layer
        c3d = C3D()
        c3d.load_state_dict(torch.load(weight_path)) # load pre-trained weight
        c3d = c3d.cuda()

        # remove c3d fc layers
        fc_removed = list(c3d.children())[:-6]

        _p3d_net = []
        relu = nn.ReLU()

        for layer in fc_removed:
            for param in layer.parameters():
                param.requires_grad = False    # no training
            if layer.__class__.__name__ == 'MaxPool3d':
                _p3d_net.extend([layer, relu]) # add activation function
            else:
                _p3d_net.append(layer)

        c3d = nn.Sequential(*_p3d_net).cuda()  # new p3d net

        self.gru = GRU(c3d)
        self.gru = self.gru.cuda()

        # # define bidirectional LSTM layer
        # self.lstm = bi_LSTM()
        # self.lstm = self.lstm.cuda()

    def forward(self, x):
        out = self.gru(x)
        return out

        # # input = [B, D, C, H, W]
        # win_start = 0
        # win_end = 48
        # win_out = []
        #
        # x = x.permute(0, 2, 1, 3, 4) # => x:[B, C, D, H, W]
        # n_snippets = 0
        #
        # # C3D layer forward (unit: 48 frames length snippet)
        # while win_end < x.shape[2]:
        #     # if n_snippets == 3:
        #     #     break           # out of memory problem
        #
        #     n_snippets += 1
        #
        #     snippet = x[:, :, win_start:win_end, :, :] # torch.Size([1, 3, 48, 256, 256])
        #     c3d_out = self.c3d(snippet) # forwarding to C3D layer
        #     c3d_out = c3d_out.squeeze() # torch.Size([512, 3, 9, 9])
        #
        #     lstm_out = self.lstm(c3d_out)
        #
        #
        #     # snippet_out = snippet_out.view(-1, 512 * 3 * 9 * 9) # flatten to 1d -> torch.Size([1, 124416])
        #     #
        #     # win_out.append(snippet_out)
        #     #
        #     # # sliding window
        #     # win_start += 6
        #     # win_end += 6
        #
        # # c3d_out = torch.cat(win_out, dim=0) # torch.Size([n_snippets=3, 124416])
        # #
        # # # bi-direactional LSTM forward
        # # lstm_out =self.lstm(c3d_out)
        # # print(lstm_out)
        # # out = torch.argmax(lstm_out, dim=1) # tensor([ 0,  0,  0,  0,  0,  1], device='cuda:0')
        # # return out


if __name__ == '__main__':
    h_loader, r_loader = get_loader(h_video_path, r_video_path)

    for idx, (frames, scores) in enumerate(r_loader):
        frames = frames.cuda()
        scores = scores.cuda()
        break

    clf = HighlightClassifier()
    out = clf(frames)
    print(out)    # 3 snippets ---- (bidirectional LSTM) ----> 6 scores
                  # [snippet 0 score-right , snippet 0 score-left, snippet 1 score-right, snippet 1 score-left,..]
