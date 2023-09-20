import argparse
import cv2
import sys
import torch
import numpy as np
from imageio import imwrite
from pathlib import Path
from tqdm import tqdm
import logging

sys.path.append(".")
import config as cfg  # noqa
from Trainer import Model  # noqa
from benchmark.utils.padder import InputPadder  # noqa


def main(parsed_args):
    logging.info("Loading model")
    TTA = True
    if parsed_args.model == "ours_small_t":
        TTA = False
        cfg.MODEL_CONFIG["LOGNAME"] = "ours_small_t"
        cfg.MODEL_CONFIG["MODEL_ARCH"] = cfg.init_model_config(
            F=16, depth=[2, 2, 2, 2, 2]
        )
    else:
        cfg.MODEL_CONFIG["LOGNAME"] = "ours_t"
        cfg.MODEL_CONFIG["MODEL_ARCH"] = cfg.init_model_config(
            F=32, depth=[2, 2, 2, 4, 4]
        )
    model = Model(-1)
    model.load_model()
    model.eval()
    model.device()
    logging.info("Model loaded. Start generating")

    path = Path(parsed_args.path)
    frames = []
    for img_path in path.glob(f"*.{parsed_args.format}"):
        frames.append(str(img_path))

    frames = sorted(frames)

    frame_counter = 0

    for frame_1, frame_2 in tqdm(zip(frames, frames[1:]), desc="Interpolating frames"):
        I0 = cv2.imread(frame_1)
        I2 = cv2.imread(frame_2)

        I0_ = (torch.tensor(I0.transpose(2, 0, 1)).cuda() / 255.0).unsqueeze(0)
        I2_ = (torch.tensor(I2.transpose(2, 0, 1)).cuda() / 255.0).unsqueeze(0)

        padder = InputPadder(I0_.shape, divisor=32)
        I0_, I2_ = padder.pad(I0_, I2_)

        images = [I0[:, :, ::-1]]
        preds = model.multi_inference(
            I0_,
            I2_,
            TTA=TTA,
            time_list=[
                (i + 1) * (1.0 / parsed_args.n) for i in range(parsed_args.n - 1)
            ],
            fast_TTA=TTA,
        )
        for pred in preds:
            images.append(
                (
                    padder.unpad(pred).detach().cpu().numpy().transpose(1, 2, 0) * 255.0
                ).astype(np.uint8)[:, :, ::-1]
            )
        images.append(I2[:, :, ::-1])
        for image in images:
            imwrite(
                f"{parsed_args.output}/{frame_counter:09d}.{parsed_args.format}", image
            )
            frame_counter += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video interpolation")
    parser.add_argument("--model", default="ours_t", type=str)
    parser.add_argument("--n", default=16, type=int)
    parser.add_argument("--path", help="Path to frames", type=str)
    parser.add_argument("--format", help="image format", type=str, default="png")
    parser.add_argument("--output", help="Path to output directory", type=str)
    parsed_args = parser.parse_args()
    assert parsed_args.model in ["ours_t", "ours_small_t"], "Model not exists!"
    main(parsed_args)
