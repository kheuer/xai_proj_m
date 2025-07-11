MAX_EPOCHS = 300
PATIENCE = 25
BATCH_SIZE = 32

params_resnet_18_random = {
    "EPOCHS": MAX_EPOCHS,
    "PATIENCE": PATIENCE,
    "BATCH_SIZE": BATCH_SIZE,
    "LEARNING_RATE": 0.001,
    "BETA_1": 0.9,
    "BETA_2": 0.999,
    "OPTIMIZER": "SGD",
    "SCHEDULER": "CosineAnnealingLR",
    "MOMENTUM": 0.53,
    "DAMPENING": 0.0145,
    "WEIGHT_DECAY": 0.0,
    # augmentation params
    "TRANSFORMATIONS_ORDER": [],
}

params_resnet_18_pretrained = {
    "EPOCHS": MAX_EPOCHS,
    "PATIENCE": PATIENCE,
    "BATCH_SIZE": BATCH_SIZE,
    "LEARNING_RATE": 0.0037,
    "BETA_1": 0.95,
    "BETA_2": 0.9999,
    "OPTIMIZER": "SGD",
    "SCHEDULER": "CosineAnnealingLR",
    "MOMENTUM": 0.736,
    "DAMPENING": 0.0327,
    "WEIGHT_DECAY": 0.0068,
    # augmentation params
    "TRANSFORMATIONS_ORDER": [],
}


params_resnet_50_pretrained = {
    "EPOCHS": MAX_EPOCHS,
    "PATIENCE": PATIENCE,
    "BATCH_SIZE": BATCH_SIZE,
    "LEARNING_RATE": 0.0059,
    "BETA_1": 0.8,
    "BETA_2": 0.99,
    "WEIGHT_DECAY": 0.0068,
    "OPTIMIZER": "SGD",
    "SCHEDULER": "StepLR",
    "MOMENTUM": 0.6991,
    "DAMPENING": 0.0713,
    "GAMMA": 0.40,
    "STEP_SIZE": 29,
    # augmentation params
    "TRANSFORMATIONS_ORDER": [],
}

params_resnet_50_random = {
    "EPOCHS": MAX_EPOCHS,
    "PATIENCE": PATIENCE,
    "BATCH_SIZE": BATCH_SIZE,
    "LEARNING_RATE": 0.0085,
    "BETA_1": 0.95,
    "BETA_2": 0.999,
    "WEIGHT_DECAY": 0.08,
    "OPTIMIZER": "SGD",
    "SCHEDULER": "LinearLR",
    "MOMENTUM": 0.81,
    "DAMPENING": 0.17,
    "GAMMA": 0.17,
    "STEP_SIZE": 47,
    # augmentation params
    "TRANSFORMATIONS_ORDER": [],
}

params_resnet_18_random_augmented = {
    "EPOCHS": MAX_EPOCHS,
    "PATIENCE": PATIENCE,
    "BATCH_SIZE": BATCH_SIZE,
    "LEARNING_RATE": 0.005183947105738152,
    "BETA_1": 0.9,
    "BETA_2": 0.9999,
    "WEIGHT_DECAY": 0.0805782454221565,
    "OPTIMIZER": "AdamW",
    "SCHEDULER": "CosineAnnealingLR",
    "MOMENTUM": 0.60506162595354,
    "DAMPENING": 0.17600007691402403,
    "GAMMA": 0.5446314874569377,
    "STEP_SIZE": 24,
    # Augmix
    "USE_AUGMIX": True,
    "SEVERITY": 1,
    "MIXTURE_WIDTH": 7,
    "CHAIN_DEPTH": 7,
    "ALPHA": 0.7480807243456533,
    "ALL_OPS": True,
    "INTERPOLATION": "NEAREST",
    # Fourier
    "USE_FOURIER": False,
    # Jigsaw
    "USE_JIGSAW": False,
    # Dlow
    "USE_DLOW": True,
    "TRANSFORMATIONS_ORDER": "Dlow,Augmix",
}


params_resnet_50_random_augmented = {
    "EPOCHS": MAX_EPOCHS,
    "PATIENCE": PATIENCE,
    "BATCH_SIZE": BATCH_SIZE,
    "LEARNING_RATE": 0.007182152751327874,
    "BETA_1": 0.95,
    "BETA_2": 0.9999,
    "WEIGHT_DECAY": 0.06805670931784548,
    "OPTIMIZER": "AdamW",
    "SCHEDULER": "None",
    "MOMENTUM": 0.8028034706965037,
    "DAMPENING": 0.05245600588394587,
    "GAMMA": 0.8567303060327642,
    "STEP_SIZE": 17,
    # Augmix
    "USE_AUGMIX": False,
    # Fourier
    "USE_FOURIER": True,
    "SQUARE_SIZE": 72,
    "ETA": 0.4767002143366463,
    # Jigsaw
    "USE_JIGSAW": False,
    # Dlow
    "USE_DLOW": False,
    "TRANSFORMATIONS_ORDER": "Augmix,Dlow,Fourier,Jigsaw",
}

params_resnet_18_pretrained_augmented = {
    "EPOCHS": MAX_EPOCHS,
    "PATIENCE": PATIENCE,
    "BATCH_SIZE": BATCH_SIZE,
    "LEARNING_RATE": 0.0023299101226650293,
    "BETA_1": 0.9,
    "BETA_2": 0.999,
    "WEIGHT_DECAY": 0.030189290982594354,
    "OPTIMIZER": "SGD",
    "SCHEDULER": "LinearLR",
    "MOMENTUM": 0.5298043637566164,
    "DAMPENING": 0.17547013593977853,
    "GAMMA": 0.3085666392448091,
    "STEP_SIZE": 33,
    # Augmix
    "USE_AUGMIX": False,
    # Fourier
    "USE_FOURIER": False,
    # Jigsaw
    "USE_JIGSAW": False,
    # Dlow
    "USE_DLOW": False,
    "TRANSFORMATIONS_ORDER": "Dlow,Augmix,Fourier,Jigsaw",
}

params_resnet_50_pretrained_augmented = {
    "EPOCHS": MAX_EPOCHS,
    "PATIENCE": PATIENCE,
    "BATCH_SIZE": BATCH_SIZE,
    "LEARNING_RATE": 0.001775604307448849,
    "BETA_1": 0.95,
    "BETA_2": 0.9999,
    "WEIGHT_DECAY": 0.043133835070306284,
    "OPTIMIZER": "SGD",
    "SCHEDULER": "ReduceLROnPlateau",
    "MOMENTUM": 0.5790938662197538,
    "DAMPENING": 0.16106357162713794,
    "GAMMA": 0.8997127883716795,
    "STEP_SIZE": 7,
    # Augmix
    "USE_AUGMIX": True,
    "SEVERITY": 5,
    "MIXTURE_WIDTH": 2,
    "CHAIN_DEPTH": 2,
    "ALPHA": 0.7216280594498236,
    "ALL_OPS": True,
    "INTERPOLATION": "BILINEAR",
    # Fourier
    "USE_FOURIER": False,
    # Jigsaw
    "USE_JIGSAW": False,
    # Dlow
    "USE_DLOW": False,
    "TRANSFORMATIONS_ORDER": "Augmix,Dlow,Fourier,Jigsaw",
}

params_resnet_18_random_augmented_art_painting = {
    "EPOCHS": MAX_EPOCHS,
    "PATIENCE": PATIENCE,
    "BATCH_SIZE": BATCH_SIZE,
    "LEARNING_RATE": 0.0010152374918409141,
    "BETA_1": 0.8,
    "BETA_2": 0.99,
    "WEIGHT_DECAY": 0.03172144159007883,
    "OPTIMIZER": "SGD",
    "SCHEDULER": "StepLR",
    "MOMENTUM": 0.5316492019026708,
    "DAMPENING": 0.16212457243756892,
    "GAMMA": 0.31413778857440355,
    "STEP_SIZE": 22,
    # Augmix
    "USE_AUGMIX": False,
    "SEVERITY": 9,
    "MIXTURE_WIDTH": 3,
    "CHAIN_DEPTH": 3,
    "ALPHA": 0.8226243334126401,
    "ALL_OPS": False,
    "INTERPOLATION": "NEAREST",
    # Fourier
    "USE_FOURIER": False,
    "SQUARE_SIZE": 39,
    "ETA": 0.8351858549305683,
    # Jigsaw
    "USE_JIGSAW": True,
    "MIN_GRID_SIZE": 4,
    "MAX_GRID_SIZE": 14,
    # Dlow
    "USE_DLOW": False,
    "TRANSFORMATIONS_ORDER": "Augmix,Dlow,Fourier,Jigsaw",
}
params_resnet_18_random_augmented_cartoon = {
    "EPOCHS": MAX_EPOCHS,
    "PATIENCE": PATIENCE,
    "BATCH_SIZE": BATCH_SIZE,
    "LEARNING_RATE": 0.007468507993702038,
    "BETA_1": 0.8,
    "BETA_2": 0.9999,
    "WEIGHT_DECAY": 0.07027279857048502,
    "OPTIMIZER": "SGD",
    "SCHEDULER": "StepLR",
    "MOMENTUM": 0.5514500584688534,
    "DAMPENING": 0.0250533094356944,
    "GAMMA": 0.4768466683660606,
    "STEP_SIZE": 37,
    # Augmix
    "USE_AUGMIX": True,
    "SEVERITY": 5,
    "MIXTURE_WIDTH": 1,
    "CHAIN_DEPTH": 2,
    "ALPHA": 0.7882899211805143,
    "ALL_OPS": True,
    "INTERPOLATION": "BILINEAR",
    # Fourier
    "USE_FOURIER": True,
    "SQUARE_SIZE": 223,
    "ETA": 0.3606188377629032,
    # Jigsaw
    "USE_JIGSAW": False,
    "MIN_GRID_SIZE": 2,
    "MAX_GRID_SIZE": 9,
    # Dlow
    "USE_DLOW": True,
    "TRANSFORMATIONS_ORDER": "Dlow,Augmix,Fourier,Jigsaw",
}

params_resnet_18_random_augmented_sketch = {
    "EPOCHS": MAX_EPOCHS,
    "PATIENCE": PATIENCE,
    "BATCH_SIZE": BATCH_SIZE,
    "LEARNING_RATE": 0.005183947105738152,
    "BETA_1": 0.9,
    "BETA_2": 0.9999,
    "WEIGHT_DECAY": 0.0805782454221565,
    "OPTIMIZER": "AdamW",
    "SCHEDULER": "CosineAnnealingLR",
    "MOMENTUM": 0.60506162595354,
    "DAMPENING": 0.17600007691402403,
    "GAMMA": 0.5446314874569377,
    "STEP_SIZE": 24,
    # Augmix
    "USE_AUGMIX": True,
    "SEVERITY": 1,
    "MIXTURE_WIDTH": 7,
    "CHAIN_DEPTH": 7,
    "ALPHA": 0.7480807243456533,
    "ALL_OPS": True,
    "INTERPOLATION": "NEAREST",
    # Fourier
    "USE_FOURIER": False,
    "SQUARE_SIZE": 118,
    "ETA": 0.3372105237773607,
    # Jigsaw
    "USE_JIGSAW": False,
    "MIN_GRID_SIZE": 4,
    "MAX_GRID_SIZE": 8,
    # Dlow
    "USE_DLOW": True,
    "TRANSFORMATIONS_ORDER": "Dlow,Fourier,Augmix,Jigsaw",
}

params_resnet_18_random_augmented_photo = {
    "EPOCHS": MAX_EPOCHS,
    "PATIENCE": PATIENCE,
    "BATCH_SIZE": BATCH_SIZE,
    "LEARNING_RATE": 0.0031317517837040586,
    "BETA_1": 0.8,
    "BETA_2": 0.999,
    "WEIGHT_DECAY": 0.07208081732080487,
    "OPTIMIZER": "SGD",
    "SCHEDULER": "CosineAnnealingLR",
    "MOMENTUM": 0.7425720410129293,
    "DAMPENING": 0.09417406688849582,
    "GAMMA": 0.43467281195705076,
    "STEP_SIZE": 12,
    # Augmix
    "USE_AUGMIX": True,
    "SEVERITY": 2,
    "MIXTURE_WIDTH": 1,
    "CHAIN_DEPTH": 7,
    "ALPHA": 0.07338540626620592,
    "ALL_OPS": True,
    "INTERPOLATION": "NEAREST",
    # Fourier
    "USE_FOURIER": True,
    "SQUARE_SIZE": 7,
    "ETA": 0.09363736846115991,
    # Jigsaw
    "USE_JIGSAW": False,
    "MIN_GRID_SIZE": 3,
    "MAX_GRID_SIZE": 11,
    # Dlow
    "USE_DLOW": False,
    "TRANSFORMATIONS_ORDER": "Dlow,Fourier,Augmix,Jigsaw",
}
