import torch
from torchvision import transforms
from fastai.vision.all import load_model

from model import Model
from extract import words_from_image, words_from_line, transform_word
from transform import Blur, Resize, ToSinglularBatch


predict_transforms = transforms.Compose(
    [
        Blur(),
        Resize(128, 32),
        transforms.ToTensor(),
        transforms.Normalize((0.9206,), (0.1546,)),
        ToSinglularBatch(),
    ]
)

img_types = {"paragraph", "line", "word"}


class PredictionEngine:
    """Class used for making predicitions using a saved model."""

    def __init__(self):
        self.model = Model()
        load_model("./models/main.pth", self.model, opt=None, with_opt=False)

    def predict_one(self, img):
        """Predict text from an image of a single word."""
        img = predict_transforms(img)

        with torch.no_grad():
            pred = self.model(img)
            word = self.model.decode(pred)

        return word[0]

    def predict_all(self, imgs):
        """Predict text from a list of single words and concatenate back into a list."""
        text = " ".join([self.predict_one(img) for img in imgs])
        return text

    def predict_image(self, img, img_type):
        """
        Perfrom correct transformation and predict text from an image,
        taking its type into account.
        """
        if img_type == "paragraph":
            to_predict = words_from_image(img)
        elif img_type == "line":
            to_predict = words_from_line(img)
        else:
            to_predict = transform_word(img)

        text = self.predict_all(to_predict)
        return text


if __name__ == "__main__":
    import cv2

    engine = PredictionEngine()
    img = cv2.imread("examples/paragraph.png")
    text = engine.predict_image(img, "paragraph")
    print(text)
