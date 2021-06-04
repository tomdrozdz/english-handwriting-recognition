import os.path

import cv2
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

from transform import Resize, Blur, ToGray


WORDS_FILE = "./data/words.txt"
WORDS_FOLDER = "./data/words"

default_transforms = transforms.Compose(
    [
        ToGray(),
        Blur(),
        Resize(128, 32),
        transforms.ToTensor(),
        transforms.Normalize((0.9206,), (0.1546,)),
    ]
)


def read_data(words_file=WORDS_FILE, words_folder=WORDS_FOLDER):
    paths_list = []
    words_list = []
    broken_iam_files = {"a01-117-05-02", "r06-022-03-05"}

    with open(words_file) as f:
        for line in f:
            if not line.startswith("#"):
                split = line.split()
                name = split[0]
                if name not in broken_iam_files:
                    full_path = os.path.join(words_folder, name + ".png")
                    paths_list.append(full_path)
                    words_list.append(split[-1])
    return paths_list, words_list


class WordDataset(Dataset):
    def __init__(self, paths, words, transform):
        self.paths = paths
        self.words = words
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path, word = self.paths[index], self.words[index]
        img = cv2.imread(path)
        img = self.transform(img)

        return img, word


def train_test_data(split=0.9, composed_transforms=None):
    if composed_transforms is None:
        composed_transforms = default_transforms

    paths, words = read_data()
    dataset = WordDataset(paths, words, composed_transforms)

    train_size = round(split * len(dataset))
    val_size = round((1 - split) * len(dataset))

    train_subset, val_subset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(0),
    )

    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, num_workers=4)

    return train_loader, val_loader


def calculate_mean_and_std():
    composed = transforms.Compose(
        [
            ToGray(),
            Blur(),
            Resize(128, 32),
            transforms.ToTensor(),
        ]
    )

    paths, words = read_data()
    dataset = WordDataset(paths, words, composed)
    loader = DataLoader(dataset, batch_size=10, num_workers=4, shuffle=False)

    mean = 0.0
    std = 0.0
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(dataset)
    std /= len(dataset)

    return mean, std


if __name__ == "__main__":
    mean, std = calculate_mean_and_std()
    print(f"Mean: {mean}, std: {std}")
