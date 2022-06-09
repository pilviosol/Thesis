import wandb

wandb.config = {
    "sample_rate": 16000,
    "n_fft": 1024,
    "hop_length": 128,
    "win_length": 1024,
    "normalization_type": "min_max_folderwise",

    "using_validation_set": True,
    "training_on_x_train_only": False,
    "batch_norm_layers": True,
    "epochs": 5000,
    "learning_rate": 0.0001,
    "batch_size": 88,
    "latent_dim": 64,
    "kl_beta": 2,

    "input_shape": (512, 256, 1),
    "conv_filters": (16, 32, 64, 128, 256),
    "conv_kernels": (3, 3, 3, 3, 3),
    "conv_strides": (2, 2, 2, 2, (2, 1)),


}

config = wandb.config

