from Levenshtein import distance  # type: ignore

from model import Model
from dataset import int_dict


class CTCWrapper:
    def __init__(self, ctc_loss_func):
        self.loss_func = ctc_loss_func

    def __call__(self, pred, args):
        targets, inputs, lengths = args
        pred = pred.log_softmax(2)

        return self.loss_func(pred, targets, inputs, lengths)


def distance_wrapper(pred, actual):
    decoded = Model.decode(pred)
    targets, _, lengths = actual

    dist = 0

    for i, pred_word in enumerate(decoded):
        actual_chars = targets.cpu().numpy()[
            0 + sum(lengths[:i]) : sum(lengths[:i]) + lengths[i]
        ]
        actual_word = "".join([int_dict[value] for value in actual_chars])
        dist += distance(pred_word, actual_word)

    dist /= sum(lengths)
    return dist


def accuracy_wrapper(pred, actual):
    decoded = Model.decode(pred)
    targets, _, lengths = actual

    words_ok = 0

    for i, pred_word in enumerate(decoded):
        actual_chars = targets.cpu().numpy()[
            0 + sum(lengths[:i]) : sum(lengths[:i]) + lengths[i]
        ]
        actual_word = "".join([int_dict[value] for value in actual_chars])
        if pred_word == actual_word:
            words_ok += 1

    accuracy = words_ok / len(decoded)

    return accuracy
