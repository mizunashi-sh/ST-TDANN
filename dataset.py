import torch
import torchvision
from torchvision.io import read_video
import os


class ContrastiveHMDB51(torch.utils.data.Dataset):
    def __init__(self, root, annotation_path, frames_per_clip, transform=None, step_between_clips=1, fold=1, train=True):
        self.base_dataset = torchvision.datasets.HMDB51(
            root=root,
            annotation_path=annotation_path,
            frames_per_clip=frames_per_clip,
            step_between_clips=step_between_clips,
            fold=fold,
            train=train,
            output_format="TCHW"
        )
        self.transform = transform
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        video, _, _ = self.base_dataset[idx]
        view1, view2 = self.transform(video)
        return view1, view2


class ContrastiveUCF101(torch.utils.data.Dataset):
    def __init__(self, root, frames_per_clip, num_workers, num_classes=400, transform=None, step_between_clips=1, split='train'):
        self.base_dataset = torchvision.datasets.UCF101(
            root=root,
            num_classes=num_classes,
            frames_per_clip=frames_per_clip,
            num_workers=num_workers,
            step_between_clips=step_between_clips,
            split=split,
            output_format='TCHW'
        )
        self.transform = transform
    
    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        video, _, _ = self.base_dataset[idx]
        view1, view2 = self.transform(video)
        return view1, view2


class ContrastiveKinetics(torch.utils.data.Dataset):
    def __init__(self, root, annotation_path, frames_per_clip, num_workers, transform=None, step_between_clips=1, fold=1, train=True):
        self.base_dataset = torchvision.datasets.Kinetics(
            root=root,
            annotation_path=annotation_path,
            frames_per_clip=frames_per_clip,
            num_workers=num_workers,
            step_between_clips=step_between_clips,
            fold=fold,
            train=train,
            output_format='TCHW'
        )
        self.transform = transform
    
    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        video, _, _ = self.base_dataset[idx]
        view1, view2 = self.transform(video)
        return view1, view2


class VideoFolderDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        
        self.video_files = []
        for dirpath, _, filenames in os.walk(self.root):
            for fname in filenames:
                if fname.endswith(('.mp4', '.avi', '.mov')):
                    self.video_files.append(os.path.join(dirpath, fname))
        
        if not self.video_files:
            raise RuntimeError(f"No video files found in {self.root}.")

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        
        # 读取视频 (返回: video, audio, info)
        video, _, info = read_video(
            video_path, 
            output_format="TCHW"
        )
        
        # 应用数据增强
        if self.transform:
            video = self.transform(video)
            
        return video, video_path  # 返回视频张量和路径
    