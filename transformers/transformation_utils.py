from typing import Callable, Union, List
from functools import reduce
import optuna
from torchvision import transforms
from torchvision.transforms import AugMix, Compose
from transformers.fourier_transformer import FourierTransformer
from transformers.jigsaw_transformer import JigsawTransform
from transformers.style_transformer import StyleTransformer
from config import MAX_EPOCHS, BATCH_SIZE, PATIENCE


def get_transform_pipeline(params: dict) -> Callable:
    transformations = []
    for fn in params["TRANSFORMATIONS_ORDER"]:
        if fn == "Augmix" and params["USE_AUGMIX"]:

            class ScaleTo255:
                def __call__(self, x):
                    return (x * 255).clamp(0, 255).byte()

            class Normalize:
                def __call__(self, x):
                    return x / 255.0

            augmix_transform = Compose(
                [
                    ScaleTo255(),
                    AugMix(
                        severity=params["SEVERITY"],
                        mixture_width=params["MIXTURE_WIDTH"],
                        chain_depth=params["CHAIN_DEPTH"],
                        alpha=params["ALPHA"],
                        all_ops=params["ALL_OPS"],
                        interpolation=params["INTERPOLATION"],
                    ),
                    Normalize(),
                ]
            )
            transformations.append(augmix_transform)

        elif fn == "Fourier" and params["USE_FOURIER"]:

            fourier_transform = transforms.Compose(
                [
                    FourierTransformer(
                        square_size=params["SQUARE_SIZE"], eta=params["ETA"]
                    )
                ]
            )
            transformations.append(fourier_transform)

        elif fn == "Jigsaw" and params["USE_JIGSAW"]:
            jigsaw_transform = transforms.Compose(
                [
                    JigsawTransform(
                        min_grid_size=params["MIN_GRID_SIZE"],
                        max_grid_size=params["MAX_GRID_SIZE"],
                    )
                ]
            )
            transformations.append(jigsaw_transform)

        elif fn == "Dlow" and params["USE_DLOW"]:
            dlow_transform = transforms.Compose(
                [
                    StyleTransformer(
                        ckpt_dir="dlow/checkpoints/",
                        target_domain=params["TARGET_DOMAIN"],
                    )
                ]
            )
            transformations.append(dlow_transform)

    # no transformations are activated
    if not transformations:
        return lambda x: x
    else:
        return lambda x: reduce(lambda acc, t: t(acc), transformations, x)
