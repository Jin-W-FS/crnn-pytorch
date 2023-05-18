import os, sys, random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from imutils import paths
from tqdm import tqdm

CHARS = [
    '', '_',
    '京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
    '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
    '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
    '新',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
    'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z',
    '学', '警', '电', '應', '挂', '使', '领', '港', '澳',
]

CHARS_DICT = {char:i for i, char in enumerate(CHARS)}

class LPRDataset(Dataset):
    CHARS = CHARS
    # CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
    # LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}

    def __init__(self, img_dir, mode, img_shape, lpr_max_len=8, PreprocFun=None, sample=-1):
        img_paths = []
        if isinstance(img_dir, str):
            img_dir = img_dir.split(';')
        for p in img_dir:
            p = os.path.expanduser(p)
            if os.path.splitext(p)[-1] in paths.image_types:
                img_paths.append(p)
            else:
                img_paths.extend(paths.list_images(p))
        random.shuffle(img_paths)
        if sample > 0 and len(img_paths) > sample:
            img_paths = random.sample(img_paths, sample)
        self.img_paths = img_paths
        self.mode = mode
        self.img_shape = tuple(img_shape) # h, w, c
        self.lpr_max_len = lpr_max_len
        if PreprocFun is not None:
            self.PreprocFun = PreprocFun
        else:
            self.PreprocFun = self.transform
        # load image data, ignore errors
        self.images = []
        pbar = tqdm(total=len(img_paths), desc=f'load {self.mode} set')
        for i, fname in enumerate(img_paths):
            try:
                data = self.loadImage(fname)
            except Exception as ex:
                print("Error loading", fname)
                continue
            self.images.append(data)
            pbar.update(1)
        pbar.close()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index]

    def loadImage(self, filename):
        if self.img_shape[-1] == 1:
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(filename)
        if img.shape != self.img_shape:
            h, w, c = self.img_shape
            if img.shape[0] != h or img.shape[1] != w:
                img = cv2.resize(img, (w, h))
            if img.shape[-1] != c:
                img = img.reshape(self.img_shape)
        img = self.PreprocFun(img)
        if self.mode == 'predict': return img

        basename = os.path.basename(filename)
        imgname, suffix = os.path.splitext(basename)
        imgname = imgname.split("-")[0].split("_")[0]
        label = self.encode(imgname)

        return img, label, len(label), filename

    @staticmethod
    def transform(img):
        img = img.astype('float32')
        img -= 127.5
        img *= 0.0078125
        img = np.transpose(img, (2, 0, 1))
        return img

    @staticmethod
    def inverse(img):
        img = img.transpose((1,2,0))
        img = 127.5 + img/0.0078125
        img = img.astype('uint8') 
        return img

    @staticmethod
    def encode(s):
        return [CHARS_DICT[c] for c in s]

    @staticmethod
    def decode(label):
        return ''.join(CHARS[int(i)] for i in label)

    @staticmethod
    def collate_fn(batch):
        imgs = []
        labels = []
        lengths = []
        fnames = []
        for img, label, length, fname in batch:
            imgs.append(torch.from_numpy(img))
            labels.extend(label)
            lengths.append(length)
            fnames.append(fname)
        return (torch.stack(imgs, 0), torch.tensor(labels), torch.tensor(lengths), fnames)
