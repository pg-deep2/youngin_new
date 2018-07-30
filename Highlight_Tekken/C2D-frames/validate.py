import torch
from dataloader import get_loader
from cnn_extractor import CNN
from torch.autograd import Variable
from vis_tool import Visualizer


model = CNN().cuda()
# model.l
model.load_state_dict(torch.load('cnn.pkl'))
print(model)

_, _, test_loader = get_loader('../Dataset/HV',
                               '../Dataset/RV',
                               '../Dataset/testRV')

test_avg_acc = 0.
test_cnt = 0

savelist = []

vis = Visualizer()



for idx, (video, label, filename) in enumerate(test_loader):

    video = video[0]
    label = label[0]
    filename = filename[0]

    start = 0
    end = 20

    out_acc= 0.
    out_predicted = []
    snp_cnt = 0

    test_cnt += 1

    while end < video.shape[0]:
        snp_cnt += 1
        snp = video[start:end, :, :]
        snp = Variable(snp).cuda()
        label_snp = label[start:end]

        # print(snp.shape)
        predicted = model(snp)  # [ frame 수, 1]


        predicted = predicted.view(1, -1)
        predicted = predicted.cpu().detach().numpy()

        predicted = predicted[0]
        label_snp = label_snp.cpu().numpy()

        # print(type(predicted), type(label))

        # gt_label_predicted_score = predicted * label_snp
        # gt_label_predicted_score = list(gt_label_predicted_score)

        # gt_label_predicted_score = gt_label_predicted_score.cpu().numpy()
        # print("Highlight frame predicted score:", gt_label_predicted_score)

        # print(gt_label_predicted_score)
        # print(gt_label_predicted_score.shape)

        # print(gt_label_predicted_score)

        # for sc in gt_label_predicted_score[0]:
        #     if sc != 0.:
        #         print("%.3f" % sc, end=' ')

        for i in range(len(predicted)):
            if predicted[i] >= 0.7:
                predicted[i] = 1.
            else:
                predicted[i] = 0.

        out_predicted.append(predicted)

        # print(type(predicted), type(label_snp))
        # label = label.cpu().numpy()
        # print(snp_cnt)
        # print(predicted)
        # print(label_snp.shape)
        acc = (predicted == label_snp).sum().item() / float(len(predicted))
        out_acc += acc

        start += 20
        end += 20


    # print("out:", snp_cnt)

    acc = out_acc / snp_cnt
    print("filename: %s accuracy: %.4f" % (filename, acc))

    test_avg_acc += acc

    savelist.append([filename, out_predicted])

    # print(out_predicted)
    # print()

test_avg_acc = test_avg_acc / test_cnt

print(savelist)

import numpy as np

print("Accuracy:", round(test_avg_acc, 4))
# vis.plot("Accuracy with test", test_avg_acc)

for f in savelist:
    np.save("./npystore/" + f[0] + ".npy", f[1])

testnp =    np.load("./npystore/testRV04(198,360).mp4.npy")
print(testnp)
print(testnp.shape)


