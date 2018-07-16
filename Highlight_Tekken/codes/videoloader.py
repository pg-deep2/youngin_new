import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2
import torch
import random
from torchvision import transforms
from PIL import Image
import functools

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Highlight Tekken video dataset
class HighlightDS(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        files = os.listdir(path) # names of files in path directory
        self.files = [os.path.join(self.path, fname) for fname in files] # abs path
        self.transforms = transform if transform is not None else lambda x: x

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        vidcap = cv2.VideoCapture(filename)

        framearr = []
        scorearr = []

        is_highlight = filename.split("/")[-2] # HV or RV

        while (vidcap.isOpened()):
            ret, frame = vidcap.read()

            if (frame is None): # end of video frames
                break

            frame = torch.from_numpy(np.asarray(frame)) # image -> numpy array -> torch tensor
            frame = frame.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
            framearr.append(frame)

            if is_highlight == 'HV': # if highlight video
                score = np.asarray([1])
            else:
                score = np.asarray([0])
            scorearr.append(torch.from_numpy(score)) # numpy score -> torch tensor
        vidcap.release()

        # list -> numpy array
        framearr = np.concatenate(framearr)
        framearr = framearr.reshape(-1, 3, 270, 480)

        scorearr = np.concatenate(scorearr)
        scorearr = scorearr.reshape(-1, 1)

        return self.transforms(framearr), scorearr

# Row(unedited) Tekken video dataset
class RowDS(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        files = os.listdir(path) # names of files in path directory
        self.files = [os.path.join(self.path, fname) for fname in files] # abs path
        self.transforms = transform if transform is not None else lambda x: x

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        vidcap = cv2.VideoCapture(filename)

        framearr = []
        scorearr = []

        is_highlight = filename.split("/")[-2]  # HV or RV

        while (vidcap.isOpened()):
            ret, frame = vidcap.read()

            if (frame is None):  # end of video frames
                break

            frame = torch.from_numpy(np.asarray(frame))  # image -> numpy array -> torch tensor
            frame = frame.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
            framearr.append(frame)

            if is_highlight == 'HV':  # if highlight video
                score = np.asarray([1])
            else:
                score = np.asarray([0])
            scorearr.append(torch.from_numpy(score))  # numpy score -> torch tensor
        vidcap.release()

        # list -> numpy array
        framearr = np.concatenate(framearr)
        framearr = framearr.reshape(-1, 3, 270, 480)

        scorearr = np.concatenate(scorearr)
        scorearr = scorearr.reshape(-1, 1)

        # Random sampling for Row videos
        sp_len = random.randint(6, 10) # 6 <= len <= 10
        sp_start = random.randint(0, len(framearr) - 120)

        framearr = framearr[sp_start:sp_start + sp_len * 12, :, :, :]
        scorearr = scorearr[sp_start:sp_start + sp_len * 12, :]

        return self.transforms(framearr), scorearr

# 멘토님 코드 참고!
def video_transform(video, image_transform):
    # apply image transform to every frame in a video
    vid = []
    for im in video:
        vid.append(image_transform(im.transpose(1,2,0)))

    vid = torch.stack(vid)
    return vid

# 멘토님 코드 참고!
def get_loader(h_path, r_path):
    image_transforms = transforms.Compose([
        Image.fromarray,
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    video_transforms = functools.partial(video_transform, image_transform=image_transforms)

    # define videos dataset
    hDS = HighlightDS(h_path, video_transforms)
    rDS = RowDS(r_path, video_transforms)

    h_loader = DataLoader(hDS, batch_size=1, drop_last=True, shuffle=True)
    r_loader = DataLoader(rDS, batch_size=1, drop_last=True, shuffle=True)

    return h_loader, r_loader

# plot video frames
def plotVideo(framearr):
    fnum = 0
    frames = framearr.numpy() # torch tensor => numpy array
    frames = frames[0]
    for f in frames:
        f = np.transpose(f, (1, 2, 0))  # (c, w, h) -> (w, h, c) for using cv2 plot format
        cv2.imshow('frame #' + str(fnum), f)
        cv2.waitKey(12)

if __name__ == "__main__":
    # get dataloaders
    h_loader, r_loader = get_loader('../dataset/HV',
                                    '../dataset/RV')

    # testing
    for idx, (frames, scores) in enumerate(h_loader):
        plotVideo(frames)
        break
