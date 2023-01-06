from PIL import Image
import cv2
import numpy as np


def dark_channel(img, size=15):
    r, g, b = cv2.split(img)
    min_img = cv2.min(r, cv2.min(g, b))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dc_img = cv2.erode(min_img, kernel)
    return dc_img


def get_atmo(img, percent=0.001):
    mean_perpix = np.mean(img, axis=2).reshape(-1)
    mean_topper = mean_perpix[: int(img.shape[0] * img.shape[1] * percent)]
    return np.mean(mean_topper)


def get_trans(img, atom, w=0.95):
    x = img / atom
    t = 1 - w * dark_channel(x, 15)
    return t


def guided_filter(p, i, r, e):
    mean_I = cv2.boxFilter(i, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    corr_I = cv2.boxFilter(i * i, cv2.CV_64F, (r, r))
    corr_Ip = cv2.boxFilter(i * p, cv2.CV_64F, (r, r))
    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p
    a = cov_Ip / (var_I + e)
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))
    q = mean_a * i + mean_b
    return q


def dehaze(im):
    im = np.array(im.convert("RGB"))
    img = im.astype("float64") / 255
    img_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype("float64") / 255
    atom = get_atmo(img)
    trans = get_trans(img, atom)
    trans_guided = guided_filter(trans, img_gray, 20, 0.0001)
    trans_guided = cv2.max(trans_guided, 0.25)
    result = np.empty_like(img)
    for i in range(3):
        result[:, :, i] = (img[:, :, i] - atom) / trans_guided + atom
    cv2.imwrite("./img_out/remove-foggy.jpg", result * 255)
    result = cv2.imread("./img_out/remove-foggy.jpg")
    return Image.fromarray(result.astype(np.uint8))


def equalist_hist(img):
    img = np.array(img.convert("RGB"))
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
    return Image.fromarray(img)
