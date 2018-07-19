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

dataset_root = 'C:/Users/young/Desktop/PROGRAPHY DATA_ver2'

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
        # scorearr = []
        #
        # is_highlight = filename.split("/")[-2] # HV or RV

        while (vidcap.isOpened()):
            ret, frame = vidcap.read()

            if (frame is None): # end of video frames
                break

            frame = torch.from_numpy(np.asarray(frame)) # image -> numpy array -> torch tensor
            frame = frame.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
            framearr.append(frame)

            # if is_highlight == 'HV': # if highlight video
            #     score = np.asarray([1])
            # else:
            #     score = np.asarray([0])
            # scorearr.append(torch.from_numpy(score)) # numpy score -> torch tensor
        vidcap.release()

        # list -> numpy array
        framearr = np.concatenate(framearr)
        framearr = framearr.reshape(-1, 3, 270, 480)

        # scorearr = np.concatenate(scorearr)
        # scorearr = scorearr.reshape(-1, 1)

        return self.transforms(framearr)

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
        # scorearr = []
        #
        # is_highlight = filename.split("/")[-2]  # HV or RV

        while (vidcap.isOpened()):
            ret, frame = vidcap.read()

            if (frame is None):  # end of video frames
                break

            frame = torch.from_numpy(np.asarray(frame))  # image -> numpy array -> torch tensor
            frame = frame.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
            framearr.append(frame)

            # if is_highlight == 'HV':  # if highlight video
            #     score = np.asarray([1])
            # else:
            #     score = np.asarray([0])
            # scorearr.append(torch.from_numpy(score))  # numpy score -> torch tensor
        vidcap.release()

        # list -> numpy array
        framearr = np.concatenate(framearr)
        framearr = framearr.reshape(-1, 3, 270, 480)

        # scorearr = np.concatenate(scorearr)
        # scorearr = scorearr.reshape(-1, 1)

        n_frames = framearr.shape[0] # num of total frames

        # Random sampling for row videos
        # 6sec <= length <= 10sec
        if n_frames >= 10 * 12 :
            snp_len = random.randint(6, 10) * 12 # num of snippet frames
        elif n_frames > 6 * 12 :
            snp_len = random.randint(6,n_frames//12) * 12
        else:
            raise IndexError("too short input video file")

        snp_start = random.randint(0, n_frames - snp_len) # snippet's start frame index

        framearr = framearr[snp_start : snp_start + snp_len, :, :, :]
        # scorearr = scorearr[snp_start : snp_start + snp_len, :]

        return self.transforms(framearr)

# Test dataset
# 멘토님 코드 참고
class TestDS(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        files = os.listdir(path)  # names of files in path directory
        self.files = [os.path.join(self.path, fname) for fname in files]  # abs path
        self.transforms = transform if transform is not None else lambda x: x

    def __getitem__(self, idx):
        filename = self.files[idx]
        print(filename)
        vidcap = cv2.VideoCapture(filename)
        framearr = []

        # set highlight frames
        videoname = os.path.split(self.files[idx])[-1]
        highlight_start = videoname.index('(')
        highlight_end = videoname.index(')')
        highlight_range = videoname[highlight_start + 1: highlight_end]  # highlight frames range

        while (vidcap.isOpened()):
            ret, frame = vidcap.read()

            if (frame is None):  # end of video frames
                break

            frame = torch.from_numpy(np.asarray(frame))  # image -> numpy array -> torch tensor
            frame = frame.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
            framearr.append(frame)

            # if is_highlight == 'HV':  # if highlight video
            #     score = np.asarray([1])
            # else:
            #     score = np.asarray([0])
            # scorearr.append(torch.from_numpy(score))  # numpy score -> torch tensor
        vidcap.release()

        # list -> numpy array
        framearr = np.concatenate(framearr)
        framearr = framearr.reshape(-1, 3, 270, 480)

        # score setting for testing
        label = np.zeros(framearr.shape[0])  # highlight label for each frames
        if ',' in highlight_range:
            start, end = highlight_range.split(',')
            label[int(start) : int(end)] = 1.

        return self.transforms(framearr), label

    def __len__(self):
        return len(self.files)


# 멘토님 코드 참고!
def video_transform(video, image_transform):
    # apply image transform to every frame in a video
    vid = []
    for im in video:
        vid.append(image_transform(im.transpose(1,2,0)))

    vid = torch.stack(vid)
    return vid

# 멘토님 코드 참고!
def get_loader(h_path, r_path, test_path):
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
    testDS = TestDS(test_path, video_transforms)

    h_loader = DataLoader(hDS, batch_size=1, drop_last=True, shuffle=True)
    r_loader = DataLoader(rDS, batch_size=1, drop_last=True, shuffle=True)
    test_loader = DataLoader(testDS, batch_size=1, drop_last=True, shuffle=False)

    return h_loader, r_loader, test_loader

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
    h_loader, r_loader, test_loader = get_loader(dataset_root + '/HV',
                                                dataset_root + '/RV',
                                                dataset_root + '/testRV')


    # # Highlight loader test
    # for idx, frames in enumerate(h_loader):
    #     plotVideo(frames)
    #     break
    #
    # # Row video loader test
    # for idx, frames in enumerate(r_loader):
    #     plotVideo(frames)
    #     break

    # # TestLoader test
    # for idx, (frames, label) in enumerate(test_loader):
    #     print(idx)
    #     plotVideo(frames)
    #     print(label)
    #     break
