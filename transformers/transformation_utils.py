from functools import reduce
import optuna
from torchvision import transforms
from torchvision.transforms import AugMix, Compose, InterpolationMode
from transformers.fourier_transformer import FourierTransformer
from transformers.jigsaw_transformer import JigsawTransform
from config import MAX_EPOCHS, BATCH_SIZE, PATIENCE


def get_transform_pipeline(params):
    # TODO: add 4th transformation and add ordering of transformations
    transformations = []
    if params["USE_AUGMIX"]:

        class ScaleTo255:
            def __call__(self, x):
                return (x * 255).clamp(0, 255).byte()

        class Normalize:
            def __call__(self, x):
                return x / 255.0

        augmix = Compose(
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
        transformations.append(augmix)

    if params["USE_FOURIER"]:
        fourier = transforms.Compose(
            [FourierTransformer(square_size=params["SQUARE_SIZE"], eta=params["ETA"])]
        )
        transformations.append(fourier)

    if params["USE_JIGSAW"]:
        jigsaw = transforms.Compose(
            [
                JigsawTransform(
                    min_grid_size=params["MIN_GRID_SIZE"],
                    max_grid_size=params["MAX_GRID_SIZE"],
                )
            ]
        )
        transformations.append(jigsaw)
    return lambda x: reduce(lambda acc, t: t(acc), transformations, x)


# TODO: complete this
def objective(trial: optuna.trial.Trial):
    params = {
        # LEARNING PARAMS
        "EPOCHS": MAX_EPOCHS,
        "PATIENCE": PATIENCE,
        "LEARNING_RATE": trial.suggest_float("LEARNING_RATE", 0.000001, 0.01),
        "BETAS": (
            trial.suggest_categorical("BETA_1", [0.8, 0.9, 0.95]),
            trial.suggest_categorical("BETA_2", [0.99, 0.999, 0.9999]),
        ),
        "WEIGHT_DECAY": trial.suggest_float("WEIGHT_DECAY", 0.0, 0.1),
        "OPTIMIZER": trial.suggest_categorical("OPTIMIZER", ["AdamW", "SGD"]),
        "SCHEDULER": trial.suggest_categorical(
            "SCHEDULER",
            ["CosineAnnealingLR", "ReduceLROnPlateau", "LinearLR", "StepLR", "None"],
        ),
        "MOMENTUM": trial.suggest_float("MOMENTUM", 0.5, 0.9),
        "DAMPENING": trial.suggest_float("DAMPENING", 0, 0.2),
        "GAMMA": trial.suggest_float("GAMMA", 0.1, 0.9),
        "STEP_SIZE": trial.suggest_int("STEP_SIZE", 5, 50),
        # TRANSFORMATION PARAMS
        # Augmix params
        "USE_AUGMIX": trial.suggest_boolean("USE_AUGMIX"),
        "SEVERITY": trial.suggest_int("SEVERITY", 1, 10),
        "MIXTURE_WIDTH": trial.suggest_int("MIXTURE_WIDTH", 1, 10),
        "CHAIN_DEPTH": trial.suggest_int("CHAIN_DEPTH", 1, 10),
        "ALPHA": trial.suggest_float("ALPHA", 0.0, 1.0),
        "ALL_OPS": trial.suggest_boolean("ALL_OPS"),
        "INTERPOLATION": trial.suggest_categorical(
            [InterpolationMode.NEAREST, InterpolationMode.BILINEAR]
        ),
        # Fourier params
        "USE_FOURIER": trial.suggest_boolean("USE_FOURIER"),
        "SQUARE_SIZE": trial.suggest_int(
            "SQUARE_SIZE_SINGLE_SIDE", 2, dataset["shape"][-1]
        ),
        "ETA": trial.suggest_float("ETA", 0, 1),
        # Jigsaw params
        "USE_JIGSAW": trial.suggest_boolean("USE_JIGSAW"),
        "MIN_GRID_SIZE": trial.suggest_int("MIN_GRID_SIZE", 2, 6),
        "MAX_GRID_SIZE": trial.suggest_int("MAX_GRID_SIZE", 6, 15),
        # Order params
        "TRANSFORMATIONS_ORDER": trial.suggest_categorical(
            "TRANSFORMATIONS_ORDER",
            [
                ("Augmix", "Dlow", "Fourier", "Jigsaw"),
                ("Augmix", "Fourier", "Dlow", "Jigsaw"),
                ("Fourier", "Augmix", "Dlow", "Jigsaw"),
                ("Fourier", "Dlow", "Augmix", "Jigsaw"),
                ("Dlow", "Augmix", "Fourier", "Jigsaw"),
                ("Dlow", "Fourier", "Augmix", "Jigsaw"),
            ],
        ),
    }
