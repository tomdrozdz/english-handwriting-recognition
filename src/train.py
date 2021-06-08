from torch import nn
from fastai.vision.all import Learner, Adam, DataLoaders, SaveModelCallback
import numpy as np

from dataset import train_test_data
from model import Model
from wrappers import CTCWrapper, distance_wrapper, accuracy_wrapper


loss_func = CTCWrapper(nn.CTCLoss(reduction="mean", zero_infinity=True))


def main():
    train, val = train_test_data()
    dataloaders = DataLoaders(train, val)
    model = Model()
    learn = Learner(
        dataloaders,
        model,
        loss_func=loss_func,
        opt_func=Adam,
        metrics=[distance_wrapper, accuracy_wrapper],
        cbs=[
            SaveModelCallback("valid_loss", fname="valid_loss_best"),
            SaveModelCallback("distance_wrapper", fname="distance_best", comp=np.less),
            SaveModelCallback(
                "accuracy_wrapper", fname="accuracy_best", comp=np.greater
            ),
        ],
    )
    # Learning rate found by using learn.lr_find()
    learn.fit_one_cycle(100, 0.0014)


if __name__ == "__main__":
    main()
