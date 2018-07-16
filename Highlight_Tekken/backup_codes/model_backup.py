import torch.nn as nn
import torch
from torch.autograd import Variable

#feature extraction에 쓰일 3D conv 모델 정의
class C3D(nn.Module):
    def __init__(self, conv_chs=16, out_chs=1):
        # args #################################
        # conv_chs: Conv3d의 output filters
        # out_chs : C3D전체의 output channel = 1
        ########################################
        super(C3D, self).__init__()

        # network
        # input size = [1, 3, 48, 256, 256] (sliding window 후에 들어와서)
        self.net = nn.Sequential(
            nn.Conv3d(3, conv_chs, (4, 4, 4), (2, 2, 2), 1, bias=False),
            nn.BatchNorm3d(conv_chs),
            nn.ReLU(),

            nn.Conv3d(conv_chs, conv_chs*2, (4, 4, 4), (2, 2, 2), 1, bias=False),
            nn.BatchNorm3d(conv_chs * 2),
            nn.ReLU(),

            nn.Conv3d(conv_chs*2, conv_chs*4, (4, 4, 4), (2, 2, 2), 1, bias=False),
            nn.BatchNorm3d(conv_chs*4),
            nn.ReLU(),

            nn.Conv3d(conv_chs*4, conv_chs*8, (4, 4, 4), (2, 2, 2), 1, bias=False),
            nn.BatchNorm3d(conv_chs*8),
            nn.ReLU(),

            # LSTM cell에 들어가려면 output channel=1 이여야 함
            nn.Conv3d(conv_chs*8, out_chs, (4, 4, 4), (2, 2, 2), 1, bias=False),
            nn.Sigmoid()

            # c3d model output size = [8, 8]
        )

    def forward(self, input):
        # input shape : [1, depth, ch, 256, 256]
        # torch docs 참고 cell input [1, ch, depth, 256, 256]으로 바꿔줘야 함
        input = input.permute(0, 2, 1, 3, 4)

        window_start = 0
        window_end = 48
        output = []
        n_snippets = 0

        while window_end < input.shape[2]:
            n_snippets += 1
            snippet = input[:, :, window_start:window_end, :, :]
            snippet_out = self.net(snippet).squeeze() # snippet_out: [8, 8]
            snippet_out = snippet_out.view(-1, 8*8)   # 1d로 펴주기
            # print(snippet_out.shape)
            # print(snippet_out)
            output.append(snippet_out)

            # sliding window
            window_start += 6
            window_end += 6

        output = torch.cat(output, dim=0) # tensor들 합치기 => 행 기준으로 아래로 차곡차곡 쌓임
                                          # output[n]=> n번째 snippet의 feature output

        # output: [n_snippets, 64]
        return output, None

# LSTM 모델 정의 many to many
# input size = [n_snippets, 64]
# output = highlight score
class LSTMcells(nn.Module):
    def __init__(self, n_classes=2, input_size=64, hidden_size=2, n_layers=1):
        super(LSTMcells, self).__init__()

        self.n_classes = n_classes
        self.n_layers = n_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    # input size = [n_snippet, 64]
    # sequence length = n_snippet
    def forward(self, input):
        # input tensor dimension 하나 증가
        input = input.unsqueeze(0)
        # print(input.shape)         # [1, n_snippets=sequence length, 64]

        n_batch = input.size(0)

        h_0 = Variable(torch.zeros(self.n_layers, n_batch, self.hidden_size))
        c_0 = Variable(torch.zeros(self.n_layers, n_batch, self.hidden_size))

        # Propagate input through lstm
        out, _ = self.lstm(input, (h_0, c_0))
        return out.view(-1, self.n_classes)  # [Pr(0), Pr(1)]이 one hot으로 나옴

class HighlightClassifier(nn.Module):
    def __init__(self, c3d, lstm):
        super(HighlightClassifier, self).__init__()
        self.conv3d = c3d
        self.lstm = lstm

    def forward(self, videoframes):
        c_out, _ = self.conv3d(videoframes)
        l_out = self.lstm(c_out)
        return torch.argmax(l_out, dim=1)