# define some directories
import torch
import torch.nn as nn
import sys

from videoloader import get_loader
from models.bidirectional import myC3D
from torch.autograd import Variable

class Test():
    def __init__(self, ckpt_path):
        _, _, test_loader = get_loader('../dataset/HV', '../dataset/RV', '../dataset/testRV')
        self.test_loader = test_loader

        self.ckpt_path = ckpt_path

        self.build_model()

    def build_model(self):
        self.c3d = myC3D().cuda()

        # if pretrained ckpt is existed
        if self.ckpt_path is not None:
            self.c3d.load_state_dict(torch.load(self.ckpt_path))

        print(self.c3d)


    # forward and compute test accuracy
    def forward_and_evaluate(self):
        self.c3d.eval() # set model eval mode
        avg_acc = 0

        for idx, (video, label) in enumerate(self.test_loader):

            acc = 0.

            video = Variable(video.cuda())
            label = label.squeeze()

            # forwarding
            predicted = self.c3d(video.cuda())
            predicted = predicted.cpu().numpy() # forwarding score만 정확도 계산에 포함

            print('Predicted output:', predicted) # [forwarding score ....., backwarding score]
            print('Predicted output length:', len(predicted))
            print('Actual label:', label)
            print('Actual label length:', len(label))

            # label => snippet 단위로 쪼개서 accuracy 계산
            start = 0
            end = 48
            snp_label = []

            while end < len(label):
                # 해당 snippet의 label에 1이 포함되어 있는 경우
                if 1. in label[start: end]:
                    snp_label.append(1)
                else:
                    snp_label.append(0)

                start += 6
                end += 6

            predicted = predicted[0: len(snp_label)]
            # print(predicted, len(predicted))
            # print(snp_label, len(snp_label))

            for pred, label in zip(predicted, snp_label):
                if pred >= 0.5 and label == 1:
                    acc += 1 / len(predicted)
                elif pred < 0.5 and label == 0:
                    acc += 1 / len(predicted)

            avg_acc += acc / len(self.test_loader)

        print("Average accuracy with test dataset:", avg_acc)

if __name__ == '__main__':
    ckpt = sys.argv[1] # cmd 첫번째 인자로 ckpt 경로 전달
    myEval = Test(ckpt)
    myEval.forward_and_evaluate()

