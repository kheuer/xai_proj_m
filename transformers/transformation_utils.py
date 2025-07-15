from typing import Callable, Union, List
from copy import deepcopy
from functools import reduce
import optuna
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import AugMix, Compose
from transformers.fourier_transformer import FourierTransformer
from transformers.jigsaw_transformer import JigsawTransform
from transformers.style_transformer import StyleTransformer
from config import MAX_EPOCHS, BATCH_SIZE, PATIENCE


def get_transform_pipeline(params: dict) -> Callable:
    collect = []
    transformations = []

    transformation_order = params["TRANSFORMATIONS_ORDER"]
    if len(transformation_order) > 0:
        transformation_order = tuple(transformation_order.split(","))

    for fn in transformation_order:
        if fn == "Augmix" and params["USE_AUGMIX"]:
            collect.append(fn)

            class ScaleTo255:
                def __call__(self, x):
                    return (x * 255).clamp(0, 255).byte()

            class Normalize:
                def __call__(self, x):
                    return x / 255.0

            if params["INTERPOLATION"] == "NEAREST":
                interpolation = InterpolationMode.NEAREST
            elif params["INTERPOLATION"] == "BILINEAR":
                interpolation = InterpolationMode.BILINEAR

            augmix = Compose(
                [
                    ScaleTo255(),
                    AugMix(
                        severity=params["SEVERITY"],
                        mixture_width=params["MIXTURE_WIDTH"],
                        chain_depth=params["CHAIN_DEPTH"],
                        alpha=params["ALPHA"],
                        all_ops=params["ALL_OPS"],
                        interpolation=(
                            InterpolationMode.BILINEAR
                            if params["INTERPOLATION"] == "bilinear"
                            else InterpolationMode.NEAREST
                        ),
                    ),
                    Normalize(),
                ]
            )
            transformations.append(augmix)

        elif fn == "Fourier" and params["USE_FOURIER"]:
            if "SQUARE_SIZE_SINGLE_SIDE" in params:
                square_size = params["SQUARE_SIZE_SINGLE_SIDE"]
            elif "SQUARE_SIZE" in params:
                square_size = params["SQUARE_SIZE"]
            else:
                square_size = None
            fourier = transforms.Compose(
                [FourierTransformer(square_size=square_size, eta=params["ETA"])]
            )
            transformations.append(fourier)

        elif fn == "Jigsaw" and params["USE_JIGSAW"]:
            collect.append(fn)
            jigsaw = transforms.Compose(
                [
                    JigsawTransform(
                        min_grid_size=params["MIN_GRID_SIZE"],
                        max_grid_size=params["MAX_GRID_SIZE"],
                    )
                ]
            )
            transformations.append(jigsaw)

        elif fn == "Dlow" and params["USE_DLOW"]:
            collect.append(fn)
            style_transform = transforms.Compose(
                [StyleTransformer("dlow/checkpoints/", params["TARGET_DOMAIN"])]
            )
            transformations.append(style_transform)

    print("setup pipeline with transformations:", collect)
    if not transformations:
        # no transformations are activated
        return lambda x: x
    else:
        return lambda x: reduce(lambda acc, t: t(acc), transformations, deepcopy(x))
