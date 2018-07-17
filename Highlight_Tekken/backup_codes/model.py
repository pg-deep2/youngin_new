import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data

from c3d import C3D
from videoloader import get_loader
from gru import GRU

h_video_path = '../../dataset/HV'
r_video_path = '../../dataset/RV'
weight_path = '../../weight/c3d.pickle'

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

    def forward(self, x):
        out = self.gru(x)
        return out


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
