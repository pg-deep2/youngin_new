# define some directories
import torch
import torch.nn as nn

from videoloader import get_loader
from models.bidirectional import C3D, GRU
from torch.autograd import Variable

class Test():
    def __init__(self, test_path, weight_path, ckpt_path):
        self.test_path = test_path
        _, _, test_loader = get_loader('../dataset/HV', '../dataset/RV', '../dataset/testRV')
        self.test_loader = test_loader

        self.weight_path = weight_path
        self.ckpt_path = ckpt_path

        self.build_model()

    def build_model(self):
        self.p3d = C3D().cuda()  # feature extraction c3d
        self.load_model()  # load pretrained weight and remove FC layers

        self.gru = GRU(self.p3d).cuda()  # bidirectional GRU
        self.gru.load_state_dict(torch.load(self.ckpt_path)) # load trained GRU ckpt

        print("MODEL:")
        print(self.gru)

    def load_model(self):
        self.p3d.load_state_dict(torch.load(self.weight_path))

        fc_removed = list(self.p3d.children())[:-6]  # remove FC layers
        _p3d_net = []
        relu = nn.ReLU().cuda()

        for layer in fc_removed:
            for param in layer.parameters():
                param.requires_grad = False
            if layer.__class__.__name__ == 'MaxPool3d':
                _p3d_net.extend([layer, relu])  # add activation function
            else:
                _p3d_net.append(layer)
        p3d_net = nn.Sequential(*_p3d_net).cuda()

        self.p3d = p3d_net

    # forward and compute test accuracy
    def forward_and_evaluate(self):
        self.gru.eval() # set model eval mode
        avg_acc = 0

        for idx, (video, label) in enumerate(self.test_loader):

            acc = 0.

            video = Variable(video.cuda())
            label = label.squeeze()

            # forwarding
            predicted = self.gru(video.cuda())
            predicted = predicted.cpu().numpy()[:len(label)] # forwarding score만 정확도 계산에 포함

            print('Predicted output:', predicted) # [forwarding score ....., backwarding score]
            print('Predicted output length:', len(predicted))
            print('Actual label:', label)
            print('Actual label length:', len(label))

            # label => snippet 단위로 쪼개서 accuracy 계산
            start = 0
            end = 48
            predicted_snp_idx = 0
            snp_label = []

            while end < len(label):
                # 해당 snippet의 label에 1이 포함되어 있는 경우
                if 1. in label[start: end]:
                    snp_label.append(1)
                else:
                    snp_label.append(0)

                start += 6
                end += 6

            print(predicted, len(predicted))
            print(snp_label, len(snp_label))

            for pred, label in zip(predicted, snp_label):
                if pred >= 0.85 and label == 1:
                    acc += 1 / len(predicted)
                elif pred < 0.85 and label == 0:
                    acc += 1 / len(predicted)

            avg_acc += acc / len(self.test_loader)

        print("Average accuracy with test dataset:", avg_acc)

if __name__ == '__main__':
    myEval = Test(test_path='../dataset/testRV',
                  weight_path='../weight/c3d.pickle',
                  ckpt_path='../checkpoints/exception/x.pth')
    myEval.forward_and_evaluate()

