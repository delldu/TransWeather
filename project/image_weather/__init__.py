"""Image Weather Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 12月 14日 星期二 00:22:28 CST
# ***
# ************************************************************************************/
#

__version__ = "1.0.0"

import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import todos

from . import weather  # for light rain
from . import restormer  # for heavy rain
from . import rain_class

import pdb


def get_tvm_model():
    """
    TVM model base on torch.jit.trace
    """
    model = weather.WeatherModel()  # Snow
    model.load_weights(model_path="models/image_desnow.pth")
    device = todos.model.get_device()
    model = model.to(device)
    model.eval()
    print(f"Running tvm model model on {device} ...")

    return model, device


class DerainModel(nn.Module):
    def __init__(self):
        super(DerainModel, self).__init__()
        self.rain_class = get_rain_class_model()
        self.remove_heavy_rain = get_heavy_rain_model()
        self.remove_light_rain = get_light_rain_model()

    def forward(self, x):
        cx = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        with torch.no_grad():
            c = self.rain_class(cx)
            _, label = torch.max(c, dim=1)
            c = label[0].item()
            if c == 0:
                x = self.remove_heavy_rain(x)
            else:
                x = self.remove_light_rain(x)
        return x


def get_rain_class_model():
    """Create rain class model."""

    model_path = "models/rain_class.pth"
    cdir = os.path.dirname(__file__)
    checkpoint = model_path if cdir == "" else cdir + "/" + model_path

    model = rain_class.MobileViT_XXS(pretrained=False)
    rain_class.RefineMobileVit(model, 2)  # only heavy, light class

    if os.path.exists(checkpoint):
        todos.model.load(model, checkpoint)

    model.eval()

    return model


def get_light_rain_model():
    """Create light model."""

    model_path = "models/remove_light_rain.pth"
    cdir = os.path.dirname(__file__)
    checkpoint = model_path if cdir == "" else cdir + "/" + model_path

    model = weather.WeatherModel()
    if os.path.exists(checkpoint):
        todos.model.load(model, checkpoint)
    # else:
    #   todos.model.load(model, "models/image_restormer.pth")
    # model = todos.model.ResizePadModel(model)

    model.eval()

    return model


def get_heavy_rain_model():
    """Create heavy rain model."""

    model_path = "models/image_restormer.pth"
    cdir = os.path.dirname(__file__)
    checkpoint = model_path if cdir == "" else cdir + "/" + model_path

    model = restormer.RestormerModel()
    if os.path.exists(checkpoint):
        todos.model.load(model, checkpoint, "params")
    # model = todos.model.ResizePadModel(model)

    model.eval()

    return model


def get_derain_model():
    """Create model."""

    model_path = "models/image_derain.pth"
    cdir = os.path.dirname(__file__)
    checkpoint = model_path if cdir == "" else cdir + "/" + model_path

    device = todos.model.get_device()
    model = DerainModel()

    # if os.path.exists(checkpoint):
    #     todos.model.load(model, checkpoint)
    # else:
    #     torch.save(model.state_dict(), checkpoint)
    todos.model.load(model, checkpoint)

    model.remove_heavy_rain = todos.model.ResizePadModel(model.remove_heavy_rain)
    model.remove_light_rain = todos.model.ResizePadModel(model.remove_light_rain)

    model = model.to(device)
    model.eval()

    print(f"Running on {device} ...")
    model = torch.jit.script(model)

    todos.data.mkdir("output")
    if not os.path.exists("output/image_derain.torch"):
        model.save("output/image_derain.torch")

    return model, device


def get_desnow_model():
    """Create model."""

    device = todos.model.get_device()
    model = weather.WeatherModel()
    model.load_weights(model_path="models/image_desnow.pth")
    model = todos.model.ResizePadModel(model)

    model = model.to(device)
    model.eval()

    print(f"Running on {device} ...")
    model = torch.jit.script(model)

    todos.data.mkdir("output")
    if not os.path.exists("output/image_desnow.torch"):
        model.save("output/image_desnow.torch")

    return model, device


def image_derain(input_files, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_derain_model()

    # load files
    image_filenames = todos.data.load_files(input_files)

    # start predict
    progress_bar = tqdm(total=len(image_filenames))
    for filename in image_filenames:
        progress_bar.update(1)

        # orig input
        input_tensor = todos.data.load_tensor(filename)
        # pytorch recommand clone.detach instead of torch.Tensor(input_tensor)
        orig_tensor = input_tensor.clone().detach()
        predict_tensor = todos.model.forward(model, device, input_tensor)
        output_file = f"{output_dir}/{os.path.basename(filename)}"

        todos.data.save_tensor([orig_tensor, predict_tensor], output_file)
    todos.model.reset_device()


def image_desnow(input_files, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_desnow_model()

    # load files
    image_filenames = todos.data.load_files(input_files)

    # start predict
    progress_bar = tqdm(total=len(image_filenames))
    for filename in image_filenames:
        progress_bar.update(1)

        # orig input
        input_tensor = todos.data.load_tensor(filename)
        # pytorch recommand clone.detach instead of torch.Tensor(input_tensor)
        orig_tensor = input_tensor.clone().detach()
        predict_tensor = todos.model.forward(model, device, input_tensor)
        output_file = f"{output_dir}/{os.path.basename(filename)}"

        todos.data.save_tensor([orig_tensor, predict_tensor], output_file)
    todos.model.reset_device()
