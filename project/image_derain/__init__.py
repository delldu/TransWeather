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

import redos
import todos

from . import light_rain
from . import heavy_rain
from . import rain_class

import pdb


class DerainModel(nn.Module):
    def __init__(self):
        super(DerainModel, self).__init__()
        self.rain_class = get_rain_class_model()
        self.remove_heavy_rain = get_heavy_rain_model()
        self.remove_light_rain = get_light_rain_model()

    def forward(self, x):
        cx = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
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

    return model


def get_light_rain_model():
    """Create light model."""

    model_path = "models/remove_light_rain.pth"
    cdir = os.path.dirname(__file__)
    checkpoint = model_path if cdir == "" else cdir + "/" + model_path

    model = light_rain.RemoveLightRainModel()
    if os.path.exists(checkpoint):
        todos.model.load(model, checkpoint)

    return model


def get_heavy_rain_model():
    """Create heavy rain model."""

    model_path = "models/remove_heavy_rain.pth"
    cdir = os.path.dirname(__file__)
    checkpoint = model_path if cdir == "" else cdir + "/" + model_path

    model = heavy_rain.RemoveHeavyRainModel()
    if os.path.exists(checkpoint):
        todos.model.load(model, checkpoint, "params")

    return model


def get_model():
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

    model = model.to(device)
    model.eval()

    print(f"Running on {device} ...")
    model = torch.jit.script(model)

    todos.data.mkdir("output")
    if not os.path.exists("output/image_derain.torch"):
        model.save("output/image_derain.torch")

    return model, device


def model_forward(model, device, input_tensor, multi_times=32):
    # zeropad for model
    H, W = input_tensor.size(2), input_tensor.size(3)
    if H % multi_times != 0 or W % multi_times != 0:
        input_tensor = todos.data.zeropad_tensor(input_tensor, times=multi_times)

    torch.cuda.synchronize()
    with torch.jit.optimized_execution(False):
        output_tensor = todos.model.forward(model, device, input_tensor)
    torch.cuda.synchronize()

    return output_tensor[:, :, 0:H, 0:W]


def image_client(name, input_files, output_dir):
    redo = redos.Redos(name)
    cmd = redos.image.Command()
    image_filenames = todos.data.load_files(input_files)
    for filename in image_filenames:
        output_file = f"{output_dir}/{os.path.basename(filename)}"
        context = cmd.weather(filename, output_file)
        redo.set_queue_task(context)
    print(f"Created {len(image_filenames)} tasks for {name}.")


def image_server(name, HOST="localhost", port=6379):
    # load model
    model, device = get_model()

    def do_service(input_file, output_file, targ):
        print(f"  derain {input_file} ...")
        try:
            input_tensor = todos.data.load_tensor(input_file)
            output_tensor = model_forward(model, device, input_tensor)
            todos.data.save_tensor(output_tensor, output_file)
            return True
        except:
            return False

    return redos.image.service(name, "image_derain", do_service, HOST, port)


def image_predict(input_files, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_model()

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
        predict_tensor = model_forward(model, device, input_tensor)
        output_file = f"{output_dir}/{os.path.basename(filename)}"

        todos.data.save_tensor([orig_tensor, predict_tensor], output_file)


def video_service(input_file, output_file, targ):
    # load video
    video = redos.video.Reader(input_file)
    if video.n_frames < 1:
        print(f"Read video {input_file} error.")
        return False

    # Create directory to store result
    output_dir = output_file[0 : output_file.rfind(".")]
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_model()

    print(f"  derain {input_file}, save to {output_file} ...")
    progress_bar = tqdm(total=video.n_frames)

    def clean_video_frame(no, data):
        # print(f"frame: {no} -- {data.shape}")
        progress_bar.update(1)

        input_tensor = todos.data.frame_totensor(data)

        # convert tensor from 1x4xHxW to 1x3xHxW
        input_tensor = input_tensor[:, 0:3, :, :]
        output_tensor = model_forward(model, device, input_tensor)

        temp_output_file = "{}/{:06d}.png".format(output_dir, no)
        todos.data.save_tensor(output_tensor, temp_output_file)

    video.forward(callback=clean_video_frame)

    redos.video.encode(output_dir, output_file)

    # delete temp files
    for i in range(video.n_frames):
        temp_output_file = "{}/{:06d}.png".format(output_dir, i)
        os.remove(temp_output_file)

    return True


def video_client(name, input_file, output_file):
    cmd = redos.video.Command()
    context = cmd.weather(input_file, output_file)
    redo = redos.Redos(name)
    redo.set_queue_task(context)
    print(f"Created 1 video tasks for {name}.")


def video_server(name, HOST="localhost", port=6379):
    return redos.video.service(name, "video_weather", video_service, HOST, port)


def video_predict(input_file, output_file):
    return video_service(input_file, output_file, None)
