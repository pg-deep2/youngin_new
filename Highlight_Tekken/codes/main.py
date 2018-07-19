"""
Usage: main.py [options] --dataroot <dataroot> --cuda
"""

import os

import random
import torch
import torch.backends.cudnn as cudnn

from config import get_config
from train import Trainer
from videoloader import get_loader

def main(config):
    if config.outf is None:
        config.outf = 'samples'
    os.system('mkdir {0}'.format(config.outf))

    config.manual_seed = random.randint(1, 10000)
    print("Random Seed: ", config.manual_seed)
    random.seed(config.manual_seed)
    torch.manual_seed(config.manual_seed)

    if config.cuda:
        torch.cuda.manual_seed_all(config.manual_seed)

    cudnn.benchmark = True

    dataroot = config.dataroot
    h_datapath = os.path.join(dataroot,"HV")
    r_datapath = os.path.join(dataroot,"RV")
    t_datapath = os.path.join(dataroot,'testRV')

    # dataroot, cache, image_size, n_channels, image_batch, video_batch, video_length):
    h_loader, r_loader, test_loader = get_loader(h_datapath, r_datapath, t_datapath)

    trainer = Trainer(config, h_loader, r_loader, test_loader)
    trainer.train()

if __name__ == "__main__":
    config = get_config()
    main(config)


"""
[1/10][1/25] - time: 17.45, h_loss: -0.000, r_loss: 27.631, total_loss: 27.631
[1/10][2/25] - time: 30.23, h_loss: -0.000, r_loss: 27.631, total_loss: 27.631
[1/10][3/25] - time: 37.07, h_loss: -0.000, r_loss: 27.631, total_loss: 27.631
[1/10][4/25] - time: 47.40, h_loss: -0.000, r_loss: 27.631, total_loss: 27.631
[1/10][5/25] - time: 64.97, h_loss: 1.727, r_loss: 9.210, total_loss: 10.937
[1/10][6/25] - time: 76.04, h_loss: -0.000, r_loss: 24.177, total_loss: 24.177
[1/10][7/25] - time: 87.90, h_loss: -0.000, r_loss: 27.631, total_loss: 27.631
[1/10][8/25] - time: 99.39, h_loss: -0.000, r_loss: 27.631, total_loss: 27.631
[1/10][9/25] - time: 110.13, h_loss: -0.000, r_loss: 17.269, total_loss: 17.269
[1/10][10/25] - time: 122.12, h_loss: -0.000, r_loss: 27.631, total_loss: 27.631
[1/10][11/25] - time: 136.25, h_loss: -0.000, r_loss: 27.631, total_loss: 27.631
[1/10][12/25] - time: 144.41, h_loss: -0.000, r_loss: 27.631, total_loss: 27.631
[1/10][13/25] - time: 157.28, h_loss: -0.000, r_loss: -0.000, total_loss: -0.000
[1/10][14/25] - time: 169.25, h_loss: -0.000, r_loss: -0.000, total_loss: -0.000
[1/10][15/25] - time: 179.97, h_loss: 23.026, r_loss: 27.631, total_loss: 50.657
[1/10][16/25] - time: 190.14, h_loss: -0.000, r_loss: 27.631, total_loss: 27.631
[1/10][17/25] - time: 203.06, h_loss: -0.000, r_loss: 11.513, total_loss: 11.513
[1/10][18/25] - time: 214.42, h_loss: -0.000, r_loss: 20.723, total_loss: 20.723
[1/10][19/25] - time: 227.00, h_loss: -0.000, r_loss: 13.816, total_loss: 13.816
[1/10][20/25] - time: 235.39, h_loss: -0.000, r_loss: 27.631, total_loss: 27.631
[1/10][21/25] - time: 246.61, h_loss: -0.000, r_loss: 27.631, total_loss: 27.631
[1/10][22/25] - time: 261.95, h_loss: 1.842, r_loss: 27.631, total_loss: 29.473
[1/10][23/25] - time: 274.35, h_loss: -0.000, r_loss: 27.631, total_loss: 27.631
[1/10][24/25] - time: 284.94, h_loss: 17.269, r_loss: 27.631, total_loss: 44.900
[1/10][25/25] - time: 293.76, h_loss: -0.000, r_loss: 27.631, total_loss: 27.631
checkpoint saved!
[2/10][1/25] - time: 309.83, h_loss: -0.000, r_loss: 27.631, total_loss: 27.631
[2/10][2/25] - time: 327.64, h_loss: -0.000, r_loss: 27.631, total_loss: 27.631
[2/10][3/25] - time: 343.13, h_loss: -0.000, r_loss: 13.816, total_loss: 13.816
[2/10][4/25] - time: 359.30, h_loss: -0.000, r_loss: 27.631, total_loss: 27.631
[2/10][5/25] - time: 368.49, h_loss: -0.000, r_loss: 27.631, total_loss: 27.631
[2/10][6/25] - time: 379.54, h_loss: -0.000, r_loss: 27.631, total_loss: 27.631
[2/10][7/25] - time: 391.64, h_loss: -0.000, r_loss: 20.723, total_loss: 20.723
[2/10][8/25] - time: 403.32, h_loss: -0.000, r_loss: -0.000, total_loss: -0.000
[2/10][9/25] - time: 418.90, h_loss: 5.526, r_loss: 27.631, total_loss: 33.157
[2/10][10/25] - time: 431.84, h_loss: -0.000, r_loss: 27.631, total_loss: 27.631
[2/10][11/25] - time: 444.07, h_loss: 27.631, r_loss: -0.000, total_loss: 27.631
[2/10][12/25] - time: 455.95, h_loss: -0.000, r_loss: 27.631, total_loss: 27.631
[2/10][13/25] - time: 463.17, h_loss: -0.000, r_loss: -0.000, total_loss: -0.000
[2/10][14/25] - time: 472.31, h_loss: -0.000, r_loss: -0.000, total_loss: -0.000
[2/10][15/25] - time: 484.46, h_loss: -0.000, r_loss: 13.816, total_loss: 13.816
[2/10][16/25] - time: 495.91, h_loss: -0.000, r_loss: 27.631, total_loss: 27.631
[2/10][17/25] - time: 507.12, h_loss: -0.000, r_loss: 27.631, total_loss: 27.631
[2/10][18/25] - time: 523.68, h_loss: 14.737, r_loss: 4.605, total_loss: 19.342
[2/10][19/25] - time: 534.75, h_loss: 27.631, r_loss: 27.631, total_loss: 55.262
[2/10][20/25] - time: 544.35, h_loss: -0.000, r_loss: 27.631, total_loss: 27.631
[2/10][21/25] - time: 554.60, h_loss: 27.631, r_loss: 20.723, total_loss: 48.354
[2/10][22/25] - time: 567.09, h_loss: 27.631, r_loss: 27.631, total_loss: 55.262
[2/10][23/25] - time: 581.65, h_loss: -0.000, r_loss: 27.631, total_loss: 27.631
[2/10][24/25] - time: 593.67, h_loss: -0.000, r_loss: 23.026, total_loss: 23.026
[2/10][25/25] - time: 610.77, h_loss: -0.000, r_loss: 27.631, total_loss: 27.631
checkpoint saved!
[3/10][1/25] - time: 621.34, h_loss: -0.000, r_loss: 27.631, total_loss: 27.631


===> 모델 잘못 짠 것 같은데 ㅠㅠㅠㅠㅠㅠㅠㅠㅠ,,, 
bidirectional.py에서 snippet score 매기는 부분을 수정해야 할 것 같은데 어떻게 해야할 지 모르겠음
지금은 forwarding / backwarding 중 한번이라도 sigmoid output이 0.5 이상이면 해당 스니펫을 1을 줌
그래서 row video들 loss가 안 줄어드나
"""