import wandb

wandb.config = {
    "sample_rate": 16000,
    "n_fft": 1024,
    "hop_length": 128,
    "win_length": 1024,
    "normalization_type": "min_max_folderwise",

    "using_validation_set": True,
    "training_on_x_train_only": False,
    "epochs": 1000,
    "learning_rate": 0.0001,
    "batch_size": 16,
    "latent_dim": 2,
    # "kl_alpha": 1000,
    "kl_beta": 0.0001,


    "input_shape": (512, 64, 1),
    "conv_filters": (512*2, 256*2, 128*2, 64*2, 32*2),
    "conv_kernels": (6, 6, 6, 6, 6),
    "conv_strides": (2, 2, 2, 2, (2, 1)),

    # "input_shape": (512, 256, 1),
    # "conv_filters": (512, 256, 128, 64, 32),
    # "conv_kernels": (3, 3, 3, 3, 3),
    # "conv_strides": (2, 2, 2, 2, (2, 1)),



    # "OVERFIT_epochs": 500,
    # "OVERFIT_learning_rate": 0.0005,
    # "OVERFIT_batch_size": 1,

}

config = wandb.config

