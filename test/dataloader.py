import numpy as np
import glob, os
import time
import cv2
import torch
from torch.utils.data import Dataset

import torchvision.transforms as transforms


class ResizeVideo(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask=None):
        if mask is None:
            for i, j in enumerate(img):
                img[i] = cv2.resize(j, tuple([self.size[1], self.size[0]]), interpolation=cv2.INTER_LINEAR)
            return img
        else:
            assert img[0].shape[:2] == mask[0].shape
            for i, (image, label) in enumerate(zip(img, mask)):
                img[i] = cv2.resize(image, tuple([self.size[1], self.size[0]]), interpolation=cv2.INTER_LINEAR)
                mask[i] = cv2.resize(label, tuple([self.size[1], self.size[0]]), interpolation=cv2.INTER_LINEAR)
        return img, mask


class ImageToTensor(object):
    def __init__(self, normalize=True):
        self.nomalize = normalize

    def __call__(self, img):
        img = torch.from_numpy(img.transpose(2, 0, 1).copy())
        if self.nomalize:
            return img.float().div(255)
        else:
            return img.float()


class TestDataset(Dataset):
    def __init__(self, test_video_dir, test_video_datasets, test_size, time_interval=1, video_time_clips=3,
                 test_all=False):
        self.test_video_dir = test_video_dir
        self.test_video_datasets = test_video_datasets
        self.time_interval = time_interval
        self.video_time_clips = video_time_clips
        self.test_size = test_size
        self.test_all = test_all

        self.dataset_name = []
        self.clips = []

        video2frames_path = {}
        for dataset in self.test_video_datasets:
            self.dataset_name += [dataset]
            dataset_path = os.path.join(self.test_video_dir, dataset)
            videos_name = sorted(os.listdir(dataset_path))
            for video in videos_name:
                video2frames_path[video] = []
                frames_path = sorted(glob.glob(os.path.join(dataset_path, video, "Imgs/*.jpg")))
                video2frames_path[video] += frames_path
        print(f"current test video dataset {'+'.join(self.dataset_name)} : {len(video2frames_path.keys())} videos")

        for video in video2frames_path.keys():
            frames = video2frames_path[video]

            div, mod = divmod(len(frames), self.video_time_clips)

            for begin in range(div):
                clip = []
                for t in range(self.video_time_clips):
                    clip.append(frames[begin * self.video_time_clips + self.time_interval * t])
                self.clips.append(clip)

            if mod != 0:
                clip = []
                for t in range(self.video_time_clips):
                    clip.append(frames[len(frames) - self.video_time_clips + t])
                self.clips.append(clip)

        print("total clips: ", len(self.clips))

        if not test_all:
            for j in self.clips[:]:
                if not (os.path.exists(
                        j[-1].replace("Imgs", "GT_object_level").replace(".jpg", ".png")) or os.path.exists(
                        j[0].replace("Imgs", "GT_object_level").replace(".jpg", ".png")) or
                        os.path.exists(
                            j[1].replace("Imgs", "GT_object_level").replace(".jpg", ".png")) or os.path.exists(
                            j[2].replace("Imgs", "GT_object_level").replace(".jpg", ".png"))):
                    self.clips.remove(j)
            print(f"you choose not test all, after clean, total clips: {len(self.clips)}")

        self.joint_transform = ResizeVideo(self.test_size)
        self.frame_transform = transforms.Compose([
            ImageToTensor(normalize=True),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def __getitem__(self, idx):
        clip_path = self.clips[idx]
        frames = []
        sizes = []
        save_paths = []

        for idx, (frame_path) in enumerate(clip_path):
            frame = cv2.imread(frame_path)
            frame = frame[:, :, ::-1]
            frames.append(frame)

            sizes.append([frame.shape[0], frame.shape[1]])

            # save_path: /DATASAET/VIDEO/xxx.png
            save_path = frame_path.split(os.path.sep)
            save_path = os.path.sep.join(save_path[-4:]).replace("Imgs" + os.path.sep, "").replace(".jpg", ".png")
            save_paths.append(save_path)

        frames = self.joint_transform(frames)
        frames_tensor = torch.zeros(len(frames), frames[0].shape[2], frames[0].shape[0], frames[0].shape[1])
        for i, j in enumerate(frames):
            frames_tensor[i, :, :, :] = self.frame_transform(j)

        return frames_tensor, torch.tensor(sizes), save_paths

    def __len__(self):
        return len(self.clips)
