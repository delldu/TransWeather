"""Model predict."""# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 12月 14日 星期二 00:22:28 CST
# ***
# ************************************************************************************/
#
import argparse
import glob
import os
import pdb  # For debug

import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
import torchvision.utils as utils
from model import get_model, model_device


def grid_image(tensor_list, nrow=3):
    grid = utils.make_grid(torch.cat(tensor_list, dim=0), nrow=nrow)
    ndarr = (
        grid.mul(255)
        .add_(0.5)
        .clamp_(0, 255)
        .permute(1, 2, 0)
        .to("cpu", torch.uint8)
        .numpy()
    )
    image = Image.fromarray(ndarr)
    return image



if __name__ == "__main__":
    """Predict."""

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--checkpoint', type=str, default="models/image_clear.pth", help="checkpint file")
    parser.add_argument(
        "--input",
        type=str,
        default="images/input/*.*",
        help="input image",
    )
    parser.add_argument("--output", type=str, default="output", help="output folder")

    args = parser.parse_args()

    # Create directory to store weights
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    model = get_model(args.checkpoint)
    device = model_device()
    model = model.to(device)
    model.eval()

    # totensor = transforms.ToTensor()
    trans_input = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trans_gt = T.Compose([T.ToTensor()])

    image_filenames = sorted(glob.glob(args.input))
    progress_bar = tqdm(total = len(image_filenames))



    for index, filename in enumerate(image_filenames):
        progress_bar.update(1)

        # GT image
        image_path = filename.replace("/input/", "/gt/")
        clean_image = Image.open(image_path).convert("RGB")
        gt_tensor = trans_gt(clean_image).unsqueeze(0).to(device)

        # orig input
        image = Image.open(filename).convert("RGB")
        orig_tensor = trans_gt(image).unsqueeze(0).to(device)

        image = Image.open(filename).convert("RGB")
        input_tensor = trans_input(image).unsqueeze(0).to(device)

        with torch.no_grad():
            predict_tensor = model(input_tensor)

        image = grid_image([orig_tensor, predict_tensor, gt_tensor], nrow=3)
        image.save("{}/image_{:02d}.png".format(args.output, index + 1))
