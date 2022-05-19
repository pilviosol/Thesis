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
    "epochs": 10000,
    "learning_rate": 0.0001,
    "batch_size": 64,
    "latent_dim": 64,
    "kl_beta": 1,


    "input_shape": (512, 64, 1),
    "conv_filters": (16*2, 32*2, 64*2, 128*2, 256*2),
    "conv_kernels": (3, 3, 3, 3, 3),
    "conv_strides": (2, 2, 2, 2, (2, 1)),




    # "OVERFIT_epochs": 500,
    # "OVERFIT_learning_rate": 0.0005,
    # "OVERFIT_batch_size": 1,

}

config = wandb.config

