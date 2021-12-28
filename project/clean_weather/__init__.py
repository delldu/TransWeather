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
import time
from tqdm import tqdm
import torch

import redos
import todos

from . import weather

import pdb

WEATHER_MEAN = (0.5, 0.5, 0.5)
WEATHER_STD = (0.5, 0.5, 0.5)
WEATHER_TIMES = 32


def get_model(checkpoint):
    """Create model."""

    model = weather.ImageWeatherModel()
    todos.model.load(model, checkpoint)
    device = todos.model.get_device()
    model = model.to(device)
    model.eval()

    return model, device


def model_forward(model, device, input_tensor):
    # normal_tensor only support CxHxW !!!
    input_tensor = input_tensor.squeeze(0)
    todos.data.normal_tensor(input_tensor, mean=WEATHER_MEAN, std=WEATHER_STD)
    input_tensor = input_tensor.unsqueeze(0)

    # zeropad for model
    H, W = input_tensor.size(2), input_tensor.size(3)
    if H % WEATHER_TIMES == 0 and H % WEATHER_TIMES == 0:
        return todos.model.forward(model, device, input_tensor)

    # else
    input_tensor = todos.data.zeropad_tensor(input_tensor, times=WEATHER_TIMES)
    output_tensor = todos.model.forward(model, device, input_tensor)
    return output_tensor[:, :, 0:H, 0:W]


def image_client(name, input_files, output_dir):
    cmd = redos.image.Command()
    redo = redos.Redos(name)

    image_filenames = todos.data.load_files(input_files)
    for filename in image_filenames:
        output_file = f"{output_dir}/{os.path.basename(filename)}"
        context = cmd.weather(filename, output_file)
        redo.set_queue_task(context)
    print(f"Created {len(image_filenames)} tasks for {name}.")


def image_server(name, HOST="localhost", port=6379):
    print(f"Start image_weather service ...")

    # load model
    checkpoint = os.path.dirname(__file__) + "/models/image_weather.pth"
    model, device = get_model(checkpoint)

    client = redos.Redos(name, HOST=HOST, port=port)

    success_count = 0
    last_runtime = time.time()
    PROCESS_IDLE_TIME = 10  # seconds
    while time.time() - last_runtime < PROCESS_IDLE_TIME:
        targ = client.get_queue_task("image_weather")
        if targ is None:
            time.sleep(0.5)
            continue

        # process task ...
        qkey = targ["key"]
        if not redos.taskarg_check(targ):
            client.set_task_state(qkey, -100)
            continue

        client.set_task_state(qkey, 0)
        input_file = redos.taskarg_search(targ, "input_file")
        output_file = redos.taskarg_search(targ, "output_file")
        print(f"  clean_weather({input_file}) ...")

        # forward
        input_tensor = todos.data.load_tensor(input_file)
        output_tensor = model_forward(model, device, input_tensor)
        todos.data.save_tensor(output_tensor, output_file)

        # update state
        client.set_task_state(qkey, 100)

        last_runtime = time.time()
        success_count = success_count + 1

    print(f"{success_count} tasks done.")
    return success_count > 0


def image_predict(input_files, output_dir, checkpoint=None):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    if checkpoint is None:
        checkpoint = os.path.dirname(__file__) + "/models/image_weather.pth"
    model, device = get_model(checkpoint)

    # load files
    image_filenames = todos.data.load_files(input_files)

    # start predict
    progress_bar = tqdm(total=len(image_filenames))
    for filename in image_filenames:
        progress_bar.update(1)

        # GT image
        # gt_filename = filename.replace("/input/", "/gt/")
        # gt_tensor = todos.data.load_tensor(gt_filename)

        # orig input
        input_tensor = todos.data.load_tensor(filename)
        # pytorch recommand clone.detach instead of torch.tensor(input_tensor)
        orig_tensor = input_tensor.clone().detach()

        predict_tensor = model_forward(model, device, input_tensor)

        output_file = f"{output_dir}/{os.path.basename(filename)}"
        # todos.data.save_tensor([orig_tensor, predict_tensor, gt_tensor], output_file)

        todos.data.save_tensor([orig_tensor, predict_tensor], output_file)


def video_client(name, input_file, output_file):
    cmd = redos.video.Command()
    redo = redos.Redos(name)
    context = cmd.weather(input_file, output_file)
    redo.set_queue_task(context)
    print(f"Created 1 video tasks for {name}.")


def video_predict(input_file, output_file, checkpoint=None):
    # load video
    video = redos.video.Reader(input_file)
    if video.n_frames < 1:
        print(f"Read video {input_file} error.")
        return False

    # Create directory to store result
    output_dir = output_file[0 : output_file.rfind(".")]
    todos.data.mkdir(output_dir)

    # load model
    if checkpoint is None:
        checkpoint = os.path.dirname(__file__) + "/models/image_weather.pth"
    model, device = get_model(checkpoint)

    print(f"Start video clean weather service {input_file}, save to {output_file} ...")
    progress_bar = tqdm(total=video.n_frames)

    def clean_video_frame(no, data):
        # print(f"frame: {no} -- {data.shape}")
        progress_bar.update(1)

        temp_tensor = todos.data.frame_totensor(data)

        input_tensor = temp_tensor[0:3, :, :].unsqueeze(0)
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


def video_server(name, HOST="localhost", port=6379):
    print(f"Start video_weather service ...")

    # load model
    checkpoint = os.path.dirname(__file__) + "/models/image_weather.pth"
    model, device = get_model(checkpoint)

    client = redos.Redos(name, HOST=HOST, port=port)
    targ = client.get_queue_task("video_weather")
    if targ is None:
        return False

    qkey = targ["key"]
    if not redos.taskarg_check(targ):
        client.set_task_state(qkey, -100)
        return False

    client.set_task_state(qkey, 0)

    input_file = redos.taskarg_search(targ, "input_file")
    output_file = redos.taskarg_search(targ, "output_file")

    ret = video_predict(input_file, output_file)

    # update state
    client.set_task_state(qkey, 100 if ret else -100)

    return ret
