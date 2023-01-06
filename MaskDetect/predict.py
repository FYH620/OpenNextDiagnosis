import argparse
from PIL import Image
from ssd import SSD
from utils.hard_weather_utils import dehaze, equalist_hist


def str2bool(v):
    return v.lower() in ("yes", "true", "1")


parser = argparse.ArgumentParser(description="Configure of predicting parameters.")
parser.add_argument(
    "--cuda", default=False, type=str2bool, help="Use CUDA to predict the result."
)
parser.add_argument(
    "--img_path",
    default="img/foggy.jpg",
    type=str,
    help="The image path for prediction.",
)
args = parser.parse_args()


if __name__ == "__main__":
    ssd = SSD(args.cuda)
    crop = False
    print("init done.")
    while True:
        info = input()
        infos = info.split(" ")
        try:
            r_image = Image.open(str(infos[0]))
        except:
            print("Open Error! Try again!")
        else:
            if int(infos[2]):
                r_image = dehaze(r_image)
            if int(infos[4]):
                r_image = equalist_hist(r_image)
            r_image = ssd.detect_image(r_image, crop=crop)
            print("one image done.")
