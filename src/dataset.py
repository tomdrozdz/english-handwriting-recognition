import os.path

import cv2
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

from transform import Resize, Blur, ToGray


WORDS_FILE = "./data/words.txt"
WORDS_FOLDER = "./data/words"
CHARS = (
    """!"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"""
)
char_dict = {char: i + 1 for i, char in enumerate(CHARS)}
int_dict = {i: char for char, i in char_dict.items()}

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
    """
    Reads data from the data folder, using the structure described in its README.
    Wokrs best with the IAM Dataset. SKips over files that are corrupted in
    the IAM Dataset. Returns two lists of file paths and labels.
    """
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
    """
    PyTorch Dataset class adapted to the custom dataset. Reads the image from given
    path every time it is called.
    """

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


def collate_fn(batch):
    """
    Custom collate_fn function for the PyTorch DataLoader. Must be used because of
    the different lengths of labels present in the dataset. Returns batches of images,
    labels encoded as integers, lengths of the inputs (always the same) and lengths
    of the labels.
    """
    imgs, words = (list(t) for t in zip(*batch))
    imgs = torch.stack(imgs, 0)

    input_lengths = torch.full((len(batch),), imgs[0].shape[1], dtype=torch.long)

    target_lengths = [len(word) for word in words]
    targets = torch.zeros(sum(target_lengths)).long()

    target_lengths = torch.tensor(target_lengths)
    for i, word in enumerate(words):
        start = sum(target_lengths[:i])
        end = target_lengths[i]
        targets[start : start + end] = torch.tensor(
            [char_dict[letter] for letter in word]
        ).long()

    return imgs, (targets, input_lengths, target_lengths)


def train_test_data(split=0.95, composed_transforms=None):
    """
    Splits data in proportions according to the split parameter and returns train and
    validation PyTorch data loaders.
    """
    if composed_transforms is None:
        composed_transforms = default_transforms

    paths, words = read_data()
    dataset = WordDataset(paths, words, composed_transforms)

    train_size = round(split * len(dataset))
    val_size = round((1 - split) * len(dataset))

    train_subset, val_subset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_subset, batch_size=50, shuffle=True, num_workers=4, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_subset, batch_size=50, num_workers=4, collate_fn=collate_fn
    )

    return train_loader, val_loader


def calculate_mean_and_std():
    """
    Function used to calculate mean and standard deviation of the images. Function was
    run before the training and the values were put into the Normalize transformation.
    """
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
