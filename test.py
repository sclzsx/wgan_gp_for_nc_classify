# Copyright 2020 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
import logging

import torch
import torchvision.utils as vutils
import wgangp_pytorch.models as models
from wgangp_pytorch.utils import create_folder
from wgangp_pytorch.utils import select_device

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="An implementation of WassersteinGAN-GP algorithm using PyTorch framework.")
    parser.add_argument("-a", "--arch", metavar="ARCH", default="lsun",
                        choices=model_names,
                        help="model architecture: " +
                             " | ".join(model_names) +
                             " (default: lsun)")
    parser.add_argument("-n", "--num-images", type=int, default=64,
                        help="How many samples are generated at one time. (default: 64).")
    parser.add_argument("--outf", default="test", type=str, metavar="PATH",
                        help="The location of the image in the evaluation process. (default: ``test``).")
    parser.add_argument("--device", default="0",
                        help="device id i.e. `0` or `0,1` or `cpu`. (default: ``cpu``).")

    args = parser.parse_args()

    print("##################################################\n")
    print("Run Testing Engine.\n")
    print(args)

    create_folder(args.outf)

    logger.info("TestEngine:")
    print("\tAPI version .......... 0.1.0")
    print("\tBuild ................ 2020.12.18-1454-f636e462")

    logger.info("Creating Testing Engine")
    device = select_device(args.device)
    model = torch.hub.load("Lornatang/WassersteinGAN-PyTorch", args.arch, progress=True, pretrained=True, verbose=False)
    model = model.to(device)

    noise = torch.randn(args.num_images, 100, 1, 1, device=device)
    with torch.no_grad():
        generated_images = model(noise)

    vutils.save_image(generated_images, f"{args.outf}/test.png", normalize=True)
    print("##################################################\n")

    logger.info("Test completed successfully.\n")
