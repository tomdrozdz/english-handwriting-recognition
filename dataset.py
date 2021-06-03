import os.path

import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from transform import Resize, Binarize, Blur, ToGray


class WordDataset(Dataset):
    def __init__(self, words_file, words_folder, transform):
        self.words_list = self.read_words_file(words_file, words_folder)
        self.transform = transform

    @staticmethod
    def read_words_file(words_file, words_folder):
        words_list = []
        broken_iam_files = ["a01-117-05-02", "r06-022-03-05"]

        with open(words_file) as f:
            for line in f:
                if not line.startswith("#"):
                    split = line.split()
                    name = split[0]
                    if name not in broken_iam_files:
                        full_path = os.path.join(words_folder, name + ".png")
                        words_list.append((full_path, split[-1]))
        return words_list

    def __len__(self):
        return len(self.words_list)

    def __getitem__(self, index):
        path, word = self.words_list[index]
        img = cv2.imread(path)
        img = self.transform(img)

        return img, word


if __name__ == "__main__":
    tf = transforms.Compose([ToGray(), Blur(), Binarize(), Resize(128, 32)])

    ds = WordDataset("./data/words.txt", "./data/words", tf)
    dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=4)
