# mypy: ignore-errors
"""
Generate models
"""

import os

import click
import torch
import numpy as np

from segment_anything.utils.transforms import (
    ResizeLongestSide,
)

from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic

import segment_anything as SAM


class Model(torch.nn.Module):
    """Model for SAM variant"""
    def __init__(self, image_size, checkpoint, model_type, device):
        super().__init__()
        self.sam = SAM.sam_model_registry[model_type](checkpoint=checkpoint)
        self.sam.to(device=device)
        self.predictor = SAM.SamPredictor(self.sam)
        self.image_size = image_size

    def forward(self, x):
        """Do forward propagation of the model

        :param x: input image
        :return: result masks and ious
        """
        self.predictor.set_torch_image(x, (self.image_size))
        return self.predictor.get_image_embedding()


@click.command()
@click.option(
    "--checkpoint",
    default="models\\pth\\sam_vit_h_4b8939.pth",
    help="path for the weight of the model",
)
@click.option(
    "--model_type",
    default="default",
    help="key to select the checkpoint corresponding model architecture",
)
@click.option(
    "--output_path",
    default="models\\vit_h_encoder_files\\vit_h_encoder.onnx",
    help="path to the output saved model with onnx format",
)
@click.option("--quantize", default=True)
@click.option("--device", default="cuda", help="device on what to run")
def main(
    checkpoint,
    model_type,
    output_path,
    quantize,
    device,
):
    """main"""
    # Target image size is 1024x720
    image_size = (1024, 1024)

    output_raw_path = output_path
    if quantize:
        # The raw directory can be deleted after the quantization is done
        output_name = os.path.basename(output_path).split(".")[0]
        output_raw_path = (
            f"{os.path.dirname(output_path)}/{output_name}_raw/"
            f"{output_name}.onnx"
        )
    os.makedirs(os.path.dirname(output_raw_path), exist_ok=True)

    sam = SAM.sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    transform = ResizeLongestSide(sam.image_encoder.img_size)

    image = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
    input_image = transform.apply_image(image)
    input_image_torch = torch.as_tensor(input_image, device=device)
    input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[
        None, :, :, :
    ]

    model = Model(image_size, checkpoint, model_type, device)
    model.eval()
    model_trace = torch.jit.trace(model, (input_image_torch))
    torch.onnx.export(
        model_trace,
        input_image_torch,
        output_raw_path,
        input_names=["image"],
        output_names=["embedding"],
        verbose=False,
        opset_version=17,
        do_constant_folding=True,
        export_params=True
    )

    if quantize:
        quantize_dynamic(
            model_input=output_raw_path,
            model_output=output_path,
            per_channel=False,
            reduce_range=False,
            weight_type=QuantType.QUInt8,
        )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
