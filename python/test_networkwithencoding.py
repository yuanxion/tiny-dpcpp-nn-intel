#!/usr/bin/env python3

# Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted
# provided that the following conditions are met:
# 	 * Redistributions of source code must retain the above copyright notice, this list of
# 	   conditions and the following disclaimer.
# 	 * Redistributions in binary form must reproduce the above copyright notice, this list of
# 	   conditions and the following disclaimer in the documentation and/or other materials
# 	   provided with the distribution.
# 	 * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
# 	   to endorse or promote products derived from this software without specific prior written
# 	   permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# @file   mlp_learning_an_image_pytorch.py
# @author Thomas Müller, NVIDIA
# @brief  Replicates the behavior of the CUDA mlp_learning_an_image.cu sample
# 		 using tiny-cuda-nn's PyTorch extension. Runs ~2x slower than native.

import argparse
import commentjson as json
import numpy as np
import os
import sys
import torch
import time

torch.manual_seed(0)
torch.use_deterministic_algorithms(True)


SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts")
sys.path.insert(0, SCRIPTS_DIR)

from common import read_image, write_image, ROOT_DIR

DATA_DIR = os.path.join(ROOT_DIR, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")

class Image(torch.nn.Module):
    def __init__(self, filename, device):
        super(Image, self).__init__()
        self.data = read_image(filename)
        self.shape = self.data.shape
        self.data = torch.from_numpy(self.data).float().to(device)

    def forward(self, xs):
        with torch.no_grad():
            # Bilinearly filtered lookup from the image. Not super fast,
            # but less than ~20% of the overall runtime of this example.
            shape = self.shape

            xs = xs * torch.tensor([shape[1], shape[0]], device=xs.device).float()
            indices = xs.long()
            lerp_weights = xs - indices.float()

            x0 = indices[:, 0].clamp(min=0, max=shape[1] - 1)
            y0 = indices[:, 1].clamp(min=0, max=shape[0] - 1)
            x1 = (x0 + 1).clamp(max=shape[1] - 1)
            y1 = (y0 + 1).clamp(max=shape[0] - 1)

            return (
                self.data[y0, x0]
                * (1.0 - lerp_weights[:, 0:1])
                * (1.0 - lerp_weights[:, 1:2])
                + self.data[y0, x1]
                * lerp_weights[:, 0:1]
                * (1.0 - lerp_weights[:, 1:2])
                + self.data[y1, x0]
                * (1.0 - lerp_weights[:, 0:1])
                * lerp_weights[:, 1:2]
                + self.data[y1, x1] * lerp_weights[:, 0:1] * lerp_weights[:, 1:2]
            )


def get_args():
    parser = argparse.ArgumentParser(
        description="Image benchmark using PyTorch bindings."
    )

    parser.add_argument(
        "image", nargs="?", default="data/images/albert.jpg", help="Image to match"
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="data/config_hash.json",
        help="JSON config for tiny-cuda-nn",
    )
    parser.add_argument(
        "n_steps",
        nargs="?",
        type=int,
        default=51, # less step to reduce saving cuda tensors
        help="Number of training steps",
    )
    parser.add_argument(
        "--device", nargs="?", default="xpu", help="Running device"
    )
    parser.add_argument(
        "result_filename", nargs="?", default="", help="Number of training steps"
    )

    args = parser.parse_args()
    return args


def pad_model_param(model_param):
    # CUDA model weights shape: 7168 = 32*32 + 64*32 + 64*64
    # XPU  model wegiths shape: 12288 = 64*64 (padded from 64*32) + 64*64 + 64*64 (padded from 64*1)

    network_input_param = model_param[:32*32]
    network_input_padding = torch.zeros(32*32*3)
    network_hidden_param = model_param[32*32:32*32+64*32]
    network_hidden_padding = torch.zeros(64*32)
    network_output_param = model_param[32*32+64*32:32*32+64*32+64*64]
    encoding_param = model_param[32*32+64*32+64*64:]

    # cat the params together with padding
    model_param_new = torch.cat((
        network_input_param, network_input_padding,
        network_hidden_param, network_hidden_padding,
        network_output_param,
        encoding_param),
        dim=-1,
    )
    return model_param_new

def save_networkwithencoding(model_param, model_param_update=None, encoding_input=None, network_output=None, targets=None, loss=None, step=-1):
    print(f'CUDA save_networkwithencoding {step = }')
    save_path = 'dump_cuda'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model_param = pad_model_param(model_param.to('cpu'))
    model_param_update = pad_model_param(model_param_update.to('cpu'))

    print(f'--> {model_param.shape = }')
    print(f'--> {network_output = }')

    save_tensor = {
        'step': step,
        'loss': loss,
        'model_param': model_param.to('cpu'),
        'model_param_update': model_param_update.to('cpu'),
        'encoding_input': encoding_input.to('cpu'),
        'network_output': network_output.to('cpu'),
        'targets': targets,
    }
    if step == -1:
        torch.save(save_tensor, f'{save_path}/networkwithencoding-step.pth')
    else:
        torch.save(save_tensor, f'{save_path}/networkwithencoding-step{step}.pth')


def load_networkwithencoding(model, loss=None, step=-1):
    print(f'XPU load_networkwithencoding {step = }')
    save_path = 'dump_cuda'

    if step == -1:
        ref_tensor = torch.load(f'{save_path}/networkwithencoding-step.pth')
    else:
        if not os.path.exists(f'{save_path}/networkwithencoding-step{step}.pth'):
            return
        ref_tensor = torch.load(f'{save_path}/networkwithencoding-step{step}.pth')

    for k,v in ref_tensor.items():
        if isinstance(v, torch.Tensor):
            ref_tensor[k] = v.to(dtype=torch.float).to(device)

    model_param_ref = ref_tensor['model_param']
    model_param_update_ref = ref_tensor['model_param_update']

    # network.reset_weights(weight_params)
    model.reset_weights(model_param_ref)

    print(f'==> compare_model_param after reset_weights, {step = }')
    all_weights = model.get_reshaped_params(datatype=model.params.data.dtype)
    model_param = model.params.data.clone()
    compare_model_param(all_weights, model_param, model_param_ref)

    batch = ref_tensor['encoding_input']
    output_ref = ref_tensor['network_output']
    targets = ref_tensor['targets']
    loss_ref = ref_tensor['loss']

    return batch, output_ref, targets, loss_ref, model_param_update_ref

def get_min_mean_max(tensor):
    t_min = tensor.min()
    t_mean = tensor.mean()
    t_max = tensor.max()

    return t_min, t_mean, t_max

def compare_model_param(all_weights, model_param, param_ref):
    # compare Network weights which has padding
    param = all_weights[0][:32, :32].clone()
    param = param.view(-1)
    start, end = 0, 1024
    diff = get_min_mean_max(torch.abs(param[:32*32] - param_ref[start:end]))
    print(f'param[{start}:{end}] {diff = }')

    param = all_weights[1][:, :32].clone()
    param = param.view(-1)
    start, end = 4096, 6144
    diff = get_min_mean_max(torch.abs(param[:64*32] - param_ref[start:end]))
    print(f'param[{start}:{end}] {diff = }')

    param = all_weights[2].clone()
    param = param.view(-1)
    start, end = 8192, 8192+2 # only compare first 2 (of all 64*1) elements because of the padding
    diff = get_min_mean_max(torch.abs(param[:2] - param_ref[start:end]))
    print(f'param[{start}:{end}] {diff = }')

    # compare Encoding weights
    start, end = 12288, 12288+1024
    diff = get_min_mean_max(torch.abs(model_param[start:end].view(-1) - param_ref[start:end]))
    print(f'param[{start}:{end}] {diff = }')
    start, end = 360000, 360000+1024	# some middle params
    diff = get_min_mean_max(torch.abs(model_param[start:end].view(-1) - param_ref[start:end]))
    print(f'param[{start}:{end}] {diff = }')
    start, end = 720656-1024, 720656
    diff = get_min_mean_max(torch.abs(model_param[start:end].view(-1) - param_ref[start:end]))
    print(f'param[{start}:{end}] {diff = }')

def compare_networkwithencoding(output, output_ref, loss, loss_ref, all_weights, model_param, model_param_ref, step=-1):
    print(f'==> compare_networkwithencoding {step = }')
    diff = get_min_mean_max(torch.abs(output - output_ref))
    print(f'output {diff = }')
    diff = get_min_mean_max(torch.abs(loss - loss_ref))
    print(f'loss {diff = }')

    print(f'==> compare_model_param after backward {step = }')
    compare_model_param(all_weights, model_param, model_param_ref)

if __name__ == "__main__":

    args = get_args()
    device = args.device

    if device == 'cuda':
        print("================================================================")
        print("This script replicates the behavior of the native CUDA example  ")
        print("mlp_learning_an_image.cu using tiny-cuda-nn's PyTorch extension.")
        print("================================================================")

        print(f"Using PyTorch version {torch.__version__} with CUDA {torch.version.cuda}")
        try:
            import tinycudann as tcnn
        except ImportError:
            print("This sample requires the tiny-cuda-nn extension for PyTorch.")
            print("You can install it by running:")
            print("============================================================")
            print("tiny-cuda-nn$ cd bindings/torch")
            print("tiny-cuda-nn/bindings/torch$ python setup.py install")
            print("============================================================")
            sys.exit()

    elif device == 'xpu':
        print("================================================================")
        print("This script replicates the behavior of the native SYCL example  ")
        print("mlp_learning_an_image.cu using tiny-dpcpp-nn's PyTorch extension.")
        print("================================================================")

        try:
            import intel_extension_for_pytorch as ipex
            import tiny_dpcpp_nn as tcnn
        except ImportError:
            print("This sample requires the tiny-dpcpp-nn extension for PyTorch.")
            print("You can install it by running:")
            print("============================================================")
            print("tiny-dpcpp-nn$ cd dpcpp_bindings")
            print("tiny-dpcpp-nn/dpcpp_bindings$ pip install -e .")
            print("============================================================")
            sys.exit()

    with open(args.config) as config_file:
        config = json.load(config_file)

    image = Image(args.image, device)
    n_channels = image.data.shape[2]

    model = tcnn.NetworkWithInputEncoding(
        n_input_dims=2,
        n_output_dims=n_channels,
        encoding_config=config["encoding"],
        network_config=config["network"],
    ).to(device)

    for name, param in model.named_parameters():
        print(f'--> {name = } {param.shape = } {param.sum() = }')

    # ===================================================================================================
    # The following is equivalent to the above, but slower. Only use "naked" tcnn.Encoding and
    # tcnn.Network when you don't want to combine them. Otherwise, use tcnn.NetworkWithInputEncoding.
    # ===================================================================================================
    # encoding = tcnn.Encoding(n_input_dims=2, encoding_config=config["encoding"])
    # encoding.to(device)
    # network = tcnn.Network(
    #     n_input_dims=encoding.n_output_dims,
    #     n_output_dims=n_channels,
    #     network_config=config["network"],
    # )
    # network.to(device)

    # model = torch.nn.Sequential(encoding, network)
    # model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Variables for saving/displaying image results
    resolution = image.data.shape[0:2]
    img_shape = resolution + torch.Size([image.data.shape[2]])
    n_pixels = resolution[0] * resolution[1]

    half_dx = 0.5 / resolution[0]
    half_dy = 0.5 / resolution[1]
    xs = torch.linspace(half_dx, 1 - half_dx, resolution[0], device=device)
    ys = torch.linspace(half_dy, 1 - half_dy, resolution[1], device=device)
    xv, yv = torch.meshgrid([xs, ys])

    xy = torch.stack((yv.flatten(), xv.flatten())).t()

    path = f"reference.jpg"
    print(f"Writing '{path}'... ", end="")
    write_image(path, image(xy).reshape(img_shape).detach().cpu().numpy())
    print("done.")

    prev_time = time.perf_counter()

    batch_size = 2**10
    interval = 1

    print(f"Beginning optimization with {args.n_steps} training steps.")

    try:
        batch = torch.rand([batch_size, 2], device=device, dtype=torch.float32)
        traced_image = torch.jit.trace(image, batch)
    except:
        # If tracing causes an error, fall back to regular execution
        print(
            f"WARNING: PyTorch JIT trace failed. Performance will be slightly worse than regular."
        )
        traced_image = image

    for i in range(args.n_steps):
        batch = torch.rand([batch_size, 2], device=device, dtype=torch.float32)
        targets = traced_image(batch)
        #encoding_output = encoding(batch)
        #output = network(encoding_output)

        print(f'Step {i} +++++++++++++++++++++++++++++++++++++++')
        if device == 'cuda':
            model_param_current = model.params.data.clone()
        elif device == 'xpu':
            batch, output_ref, targets, loss_ref, model_param_update_ref = load_networkwithencoding(model, step=i)

        output = model(batch)

        # if i == 1:
        #     exit()
        relative_l2_error = (output - targets.to(output.dtype)) ** 2 / (
            output.detach() ** 2 + 0.01
        )
        loss = relative_l2_error.mean()

        # print("enc params: ", model.params[64 * 64 * 3 :])
        # print("enc params min: ", model.params[64 * 64 * 3 :].min())
        # print("enc params max: ", model.params[64 * 64 * 3 :].max())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        model_param_update = model.params.data.clone()
        if device == 'cuda':
            save_networkwithencoding(model_param_current, model_param_update=model_param_update, encoding_input=batch, network_output=output, targets=targets.to('cpu'), loss=loss_val, step=i)
        elif device == 'xpu':
            all_weights = model.get_reshaped_params(datatype=model.params.data.dtype)
            compare_networkwithencoding(output, output_ref, loss, loss_ref, all_weights, model_param_update, model_param_update_ref, step=i)

        # print("After enc params: ", model.params[64 * 64 * 3 :])
        # print("After enc params min: ", model.params[64 * 64 * 3 :].min())
        # print("After enc params max: ", model.params[64 * 64 * 3 :].max())
        if i % interval == 0:
            loss_val = loss.item()
            if device == 'cuda':
                torch.cuda.synchronize()
            elif device == 'xpu':
                torch.xpu.synchronize()
            elapsed_time = time.perf_counter() - prev_time
            print(f"Step#{i}: loss={loss_val} time={int(elapsed_time*1000000)}[µs]")

            path = f"{i}.jpg"
            print(f"Writing '{path}'... ", end="")
            with torch.no_grad():
                write_image(
                    path,
                    model(xy).reshape(img_shape).clamp(0.0, 1.0).detach().cpu().numpy(),
                )
            print("done.")

            # Ignore the time spent saving the image
            prev_time = time.perf_counter()

            if i > 0 and interval < 1000:
                interval *= 10

    # save_encoding_file("grid", config, encoding, batch_size, input_dims=2, name="image")
    # encoding_param = encoding.params.detach().cpu().numpy()
    # network_param = network.params.detach().cpu().numpy()

    # np.savetxt("encoding_params.csv", encoding_param, delimiter=",")
    # np.savetxt("network_params.csv", network_param, delimiter=",")
    # np.savetxt("encoding_input.csv", batch.detach().cpu().numpy(), delimiter=",")
    # np.savetxt("encoding_output.csv", encoding(batch).detach().cpu().numpy(), delimiter=",")
    # np.savetxt("network_output.csv", output.detach().cpu().numpy(), delimiter=",")

    if args.result_filename:
        print(f"Writing '{args.result_filename}'... ", end="")
        with torch.no_grad():
            write_image(
                args.result_filename,
                model(xy).reshape(img_shape).clamp(0.0, 1.0).detach().cpu().numpy(),
            )
        print("done.")

    if device == 'cuda':
        tcnn.free_temporary_memory()
