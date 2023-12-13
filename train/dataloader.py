import random

import numpy as np
import glob, os
import time
import cv2
import torch
from torch.utils.data import Dataset

import torchvision.transforms as transforms
class JointCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *img):
        for t in self.transforms:
            img = t(*img)
        return img
class GrayImageToTensor(object):
    def __init__(self, normalize=True):
        self.nomalize = normalize
    def __call__(self, img):
        img = np.expand_dims(img, 2)
        img = torch.from_numpy(img.transpose(2,0,1).copy())
        if self.nomalize:
            return img.float().div(255)
        else:
            return img.float()

class GrayVideoToTensor(object):
    def __init__(self, normalize=True):
        self.nomalize = normalize
    def __call__(self, video):
        video_tensor = torch.zeros(len(video),1,*(video[0].shape))
        for i, j in enumerate(video):

            img = np.expand_dims(j, 2)
            img = torch.from_numpy(img.transpose(2,0,1).copy())
            if self.nomalize:
                video_tensor[i, :, :, :] = img.float().div(255)
            else:
                video_tensor[i, :, :, :] = img.float()
        return video_tensor
class RandomHorizontallyFlipImage(object):
    def __call__(self, img, mask=None):
        if mask is None:
            if random.random() < 0.5:
                return img[:,::-1,:]
            return img
        else:
            assert img.shape[:2] == mask.shape
            if random.random() < 0.5:
                return img[:, ::-1, :], mask[:, ::-1]
            return img, mask
class RandomHorizontallyFlipVideo(object):
    def __call__(self, img, mask=None):
        if mask is None:
            if random.random() < 0.5:
                for index,i in enumerate(img):
                    img[index] = i[:, ::-1, :]
                return img
            return img
        else:
            assert img[0].shape[:2] == mask[0].shape
            if random.random() < 0.5:
                for index, (i, j) in enumerate(zip(img, mask)):
                    img[index] = i[:, ::-1, :]
                    mask[index] = j[:, ::-1]
            return img, mask
class RandomFlipVideo(object):
    def __call__(self, img, mask=None):
        if mask is None:
            if random.random() < 0.5:
                return list(reversed(img))
            return img
        else:
            assert img[0].shape[:2] == mask[0].shape
            if random.random() < 0.5:
                return list(reversed(img)),list(reversed(mask))

            return img, mask
class ResizeImage(object):
    def __init__(self, size, gt_resize_type=cv2.INTER_NEAREST):
        self.size = size
        self.gt_resize_type = gt_resize_type
    def __call__(self, img, mask=None):

        if mask is None:
            img = cv2.resize(img, tuple([self.size[1], self.size[0]]), interpolation=cv2.INTER_LINEAR)
            return img
        else:
            assert img.shape[:2] == mask.shape
            img = cv2.resize(img, tuple([self.size[1], self.size[0]]), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, tuple([self.size[1], self.size[0]]), interpolation=cv2.INTER_LINEAR)
        return img, mask
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

class ImageTrainDataset(Dataset):
    def __init__(self, train_image_dir, train_image_datasets, train_size):
        #images : /path/to/train/DATASET/VIDEO_NAME/Imgs/*.jpg
        #labels : /path/to/train/DATASET/VIDEO_NAME/GT_object_level/*.png
        self.train_image_dir = train_image_dir
        self.train_image_datasets = train_image_datasets
        self.train_size = train_size

        #record all train samples
        self.images = []

        self.dataset_name = []
        for dataset in self.train_image_datasets:
            self.dataset_name += [dataset]
            self.images += glob.glob(os.path.join(self.train_image_dir, dataset,r"*/Imgs/*.jpg"))
        print(f"current image dataset {'+'.join(self.dataset_name)} : {len(self.images)} images")

        self.joint_transform = JointCompose([
            ResizeImage(train_size),
            RandomHorizontallyFlipImage(),
        ])
        self.image_transform = transforms.Compose([
            ImageToTensor(normalize=True),
            transforms.Normalize(mean = (0.485, 0.456, 0.406), std =(0.229, 0.224, 0.225)),
        ])
        self.label_transform = transforms.Compose([
            GrayImageToTensor(normalize=True)
        ])

    def __getitem__(self, idx):
        path = self.images[idx]
        label = path.replace("Imgs", "GT_object_level").replace(".jpg", ".png")

        image = cv2.imread(path)
        image = image[:,:,::-1] #RGB
        label = cv2.imread(label, cv2.IMREAD_GRAYSCALE)

        #训练时候翻转
        image, label = self.joint_transform(image, label)

        image = self.image_transform(image)
        label = self.label_transform(label)

        return image, label

    def __len__(self):
        return len(self.images)
class VideoTrainDataset(Dataset):
    def __init__(self,train_video_dir, train_video_datasets, train_size, time_interval=1,video_time_clips = 3):
        self.train_video_dir = train_video_dir
        self.train_video_datasets = train_video_datasets
        self.time_interval = time_interval
        self.video_time_clips = video_time_clips
        self.train_size = train_size


        self.dataset_name = []
        self.clips = []

        #video2frames_path: {video_name:[all, of, the frames]}
        video2frames_path = {}
        for dataset in self.train_video_datasets:
            self.dataset_name += [dataset]
            dataset_path = os.path.join(self.train_video_dir, dataset)
            videos_name = sorted(os.listdir(dataset_path))
            for video in videos_name:
                video2frames_path[video] = []
                frames_path = sorted(glob.glob(os.path.join(dataset_path,video,"Imgs/*.jpg")))
                video2frames_path[video] += frames_path
        print(f"current video dataset {'+'.join(self.dataset_name)} : {len(video2frames_path.keys())} videos")

        for video in video2frames_path.keys():
            frames = video2frames_path[video]
            for begin in range(len(frames)-(self.video_time_clips-1)*self.time_interval):
                clip = []
                for t in range(self.video_time_clips):
                    clip.append(frames[begin + self.time_interval*t])
                self.clips.append(clip)

        self.joint_transform = JointCompose([
            ResizeVideo(self.train_size),
            RandomHorizontallyFlipVideo(),
            RandomFlipVideo(),
        ])
        self.labels_transform = GrayVideoToTensor(normalize=True)

        self.frame_transform = transforms.Compose(
            [
                ImageToTensor(normalize=True),
                transforms.Normalize(mean = (0.485, 0.456, 0.406), std =(0.229, 0.224, 0.225))
            ])

    def __getitem__(self, idx):
        clip_path = self.clips[idx]
        frames = []
        labels = []
        for idx, (frame_path) in enumerate(clip_path):
            frame = cv2.imread(frame_path)
            frame = frame[:, :, ::-1]
            label = cv2.imread(frame_path.replace("Imgs", "GT_object_level").replace(".jpg", ".png"), cv2.IMREAD_GRAYSCALE)
            frames.append(frame)
            labels.append(label)

        frames, labels = self.joint_transform(frames, labels)

        labels_tensor = self.labels_transform(labels)

        frames_tensor = torch.zeros(len(frames), frames[0].shape[2],frames[0].shape[0],frames[0].shape[1])
        for i, j in enumerate(frames):
            # T C H W
            frames_tensor[i, :, :, :] = self.frame_transform(j)
        return frames_tensor, labels_tensor

    def __len__(self):
        return len(self.clips)


