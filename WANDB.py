import wandb

wandb.config = {
    "sample_rate": 16000,
    "n_fft": 1024,
    "hop_length": 128,
    "win_length": 1024,
    "normalization_type": "min_max_folderwise",

    "using_validation_set": True,
    "training_on_x_train_only": False,
    "epochs": 105,
    "learning_rate": 0.0005,
    "batch_size": 32,
    "latent_dim": 16,
    "kl_alpha": 1,

    "input_shape": (512, 256, 1),
    "conv_filters": (512, 256, 128, 64, 32),
    "conv_kernels": (3, 3, 3, 3, 3),
    "conv_strides": (2, 2, 2, 2, (2, 1)),

    # "OVERFIT_epochs": 17000,
    # "OVERFIT_learning_rate": 0.001,
    # "OVERFIT_batch_size": 1,

}

config = wandb.config

