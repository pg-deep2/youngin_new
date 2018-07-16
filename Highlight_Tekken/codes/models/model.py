import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data

from c3d import C3D
from videoloader import get_loader
from lstm import bi_LSTM

# c3d layer + bidirectional-LSTM layer
class HighlightClassifier(nn.Module):
    def __init__(self):
        super(HighlightClassifier, self).__init__()
        # define C3D layer
        self.c3d = C3D()
        self.c3d.load_state_dict(torch.load('../../pretrained-weight/c3d.pickle')) # load pre-trained weight
        self.c3d = self.c3d.cuda()

        # remove c3d fc layers
        fc_removed = list(self.c3d.children())[:-6]
        self.c3d = nn.Sequential(*fc_removed)

        # define bidirectional LSTM layer
        self.lstm = bi_LSTM()
        self.lstm = self.lstm.cuda()

    def forward(self, x):
        # input = [B, D, C, H, W]
        win_start = 0
        win_end = 48
        win_out = []

        x = x.permute(0, 2, 1, 3, 4) # => x:[B, C, D, H, W]
        n_snippets = 0

        # C3D layer forward (unit: 48 frames length snippet)
        while win_end < x.shape[2]:
            if n_snippets == 3:
                break           # out of memory problem

            print("Snippet #", n_snippets)
            n_snippets += 1
            snippet = x[:, :, win_start:win_end, :, :]
            print("C3D input snippet shape:", snippet.shape)  # torch.Size([1, 3, 48, 256, 256])

            snippet_out = self.c3d(snippet) # forwarding to C3D layer
            print("Before squeeze, one snippet C3D output shape:", snippet_out.shape) # torch.Size([1, 512, 3, 9, 9])

            snippet_out = snippet_out.squeeze()
            print("After squeeze, one snippet C3D output shape:", snippet_out.shape) # torch.Size([512, 3, 9, 9])

            snippet_out = snippet_out.view(-1, 512 * 3 * 9 * 9) # flatten to 1d
            print("After reshape, one snippet C3D output shape:", snippet_out.shape) # torch.Size([1, 124416])

            win_out.append(snippet_out)

            # sliding window
            win_start += 6
            win_end += 6
            print("\n\n")

        c3d_out = torch.cat(win_out, dim=0)
        print("all snippets C3D output shape:", c3d_out.shape) # torch.Size([n_snippets=3, 124416])

        # bi-direactional LSTM forward
        lstm_out =self.lstm(c3d_out)
        print(lstm_out)
        out = torch.argmax(lstm_out, dim=1) # tensor([ 0,  0,  0,  0,  0,  1], device='cuda:0')
        return out


if __name__ == '__main__':
    h_loader, r_loader = get_loader('../../dataset/HV', '../../dataset/RV')

    for idx, (frames, scores) in enumerate(r_loader):
        frames = frames.cuda()
        scores = scores.cuda()
        break

    clf = HighlightClassifier()
    out = clf(frames)
    print(out)    # 3 snippets ---- (bidirectional LSTM) ----> 6 scores
                  # [snippet 0 score-right , snippet 0 score-left, snippet 1 score-right, snippet 1 score-left,..]
