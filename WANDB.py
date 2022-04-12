import wandb

wandb.config = {
    "sample_rate": 16000,
    "n_fft": 1024,
    "hop_length": 128,
    "win_length": 1024,
    # "bins_per_octave": 48,
    # "num_octaves": 8,
    # "n_iter": 1,
    # "n_units": 2048,
    # "kl_beta": 0.0005,
    # "batch_norm": True,
    # "output_activation": "relu",

    "using_validation_set": True,
    "training_on_x_train_only": False,
    "normalization_type": "min_max_folderwise",
    "epochs": 80,
    "learning_rate": 0.001,
    "batch_size": 32,

    "latent_dim": 16,
    "kl_alpha": 10,
    "OVERFIT_epochs": 17000,
    "OVERFIT_learning_rate": 0.001,
    "OVERFIT_batch_size": 1,

    # "buffer_size": 60000,
    # "buffer_size_dataset": True,
    # "continue_training": False,
    # "max_ckpts_to_keep": 2,
    # "checkpoint_epochs": 15,
    # "save_best_only": True,
    # "learning_schedule": False,
    # "early_patience_epoch": 50,
    # "early_delta": 1e-9,
    # "adam_beta_1": 0.9,
    # "adam_beta_2": 0.999
}

config = wandb.config

