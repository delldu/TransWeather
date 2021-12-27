"""Image Weather Package."""# coding=utf-8
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

import redos
import todos

from . import weather

import pdb


def get_model(checkpoint):
    """Create model."""

    model = weather.ImageWeatherModel()
    todos.model.load(model, checkpoint)
    return model


def client(name, input_files, output_dir):
    cmd = redos.image.Command()
    redo = redos.Redos(name)

    image_filenames = todos.data.load_files(input_files)
    for filename in image_filenames:
        output_file = f"{output_dir}/{os.path.basename(filename)}"
        context = cmd.weather(filename, output_file)
        redo.set_queue_task(context)
    print(f"Created {len(image_filenames)} tasks for {name}.")


def service(name, HOST='localhost', port=6379):
    """image weather service"""

    print(f"Start image weather service, pid is {os.getpid()}...")

    # load model
    checkpoint = os.path.dirname(__file__) + "/models/image_weather.pth"
    model = get_model(checkpoint)
    device = todos.model.get_device()
    model = model.to(device)
    model.eval()

    client = redos.Redos(name, HOST=HOST, port=port)

    success_count = 0
    PROCESS_IDLE_TIME = 10  # seconds
    last_runtime = time.time()
    while time.time() - last_runtime < PROCESS_IDLE_TIME:
        task = client.get_queue_task("image_weather")
        if task is None:
            time.sleep(1)
            continue

        targ = redos.taskarg_parse(task["content"])
        if targ is None:
            print("taskarg_parse error for image_clean task !!!")
            # task queue is not valid, continue
            continue

        # process task ...
        qkey = targ["key"]
        client.set_task_state(qkey, 0)

        input_file = redos.taskarg_search(targ, "input_file")
        output_file = redos.taskarg_search(targ, "output_file")

        # forward
        input_tensor = todos.data.load_tensor(input_file)
        input_tensor = todos.data.normal_tensor(input_tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        output_tensor = todos.model.forward(model, device, input_tensor)

        todos.data.save_tensor(output_tensor.cpu(), output_file)

        # update state
        client.set_task_state(qkey, 100)

        last_runtime = time.time()
        success_count = success_count + 1

    print(f"{success_count} tasks done.")
    return success_count > 0


def predict(input_files, output_dir, checkpoint=None):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    if checkpoint is None:
        checkpoint = os.path.dirname(__file__) + "/models/image_weather.pth"
    model = get_model(checkpoint)
    device = todos.model.get_device()
    model = model.to(device)
    model.eval()

    # load files
    image_filenames = todos.data.load_files(input_files)

    # start predict
    progress_bar = tqdm(total = len(image_filenames))
    for filename in image_filenames:
        progress_bar.update(1)

        # GT image
        gt_filename = filename.replace("/input/", "/gt/")
        gt_tensor = todos.data.load_tensor(gt_filename)

        # orig input
        orig_tensor = todos.data.load_tensor(filename)

        input_tensor = todos.data.normal_tensor(orig_tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        predict_tensor = todos.model.forward(model, device, input_tensor)

        image = todos.data.grid_image([orig_tensor.cpu(), predict_tensor.cpu(), gt_tensor.cpu()], nrow=3)

        output_file = f"{output_dir}/{os.path.basename(filename)}"
        image.save(output_file)

