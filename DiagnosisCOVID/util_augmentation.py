from imgaug import augmenters as iaa
from numpy import random
from math import sqrt
from config import is_random_erasing


class RandomErasing(object):
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, img):
        if random.uniform(0, 1) >= self.probability:
            return img
        for _ in range(100):
            area = img.shape[0] * img.shape[1]
            target_area = random.uniform(0.02, 0.4) * area
            aspect_ratio = random.uniform(0.3, 1 / 0.3)
            height = int(round(sqrt(target_area * aspect_ratio)))
            width = int(round(sqrt(target_area / aspect_ratio)))

            if width < img.shape[1] and height < img.shape[0]:
                top = random.randint(0, img.shape[0] - height)
                left = random.randint(0, img.shape[1] - width)
                img[top : top + height, left : left + width, :] = random.randint(
                    255, size=(height, width, 3)
                )
                return img
        return img


class DiagnosisAugmentation(object):
    def __init__(self, size, mean):
        self.train_seq = iaa.Sequential(
            [
                iaa.Fliplr(0.5),
                iaa.Sometimes(
                    0.5,
                    iaa.Affine(
                        scale={"x": (0.5, 1.5), "y": (0.5, 1.5)},
                        translate_percent={"x": (-0.1, 0.1), "y": (-0.2, 0.2)},
                        rotate=(-45, 45),
                    ),
                ),
                iaa.Sometimes(0.5, iaa.contrast.LinearContrast()),
                iaa.Sometimes(0.5, iaa.Sharpen()),
                iaa.Sometimes(0.5, iaa.Multiply()),
                iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise()),
                iaa.Resize(size),
            ]
        )
        self.random_erasing = RandomErasing(probability=0.2)
        self.mean = mean

    def __call__(self, img):
        img = self.train_seq.augment_image(img)
        if is_random_erasing:
            img = self.random_erasing(img)
        img -= self.mean
        return img


class BaseTransform(object):
    def __init__(self, size, mean):
        self.test_seq = iaa.Sequential([iaa.Resize(size)])
        self.mean = mean

    def __call__(self, img):
        img = self.test_seq.augment_image(img)
        img -= self.mean
        return img
