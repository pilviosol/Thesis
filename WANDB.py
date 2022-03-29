import wandb

wandb.config = {
    "sample_rate": 44100,
    "hop_length": 128,
    "bins_per_octave": 48,
    "num_octaves": 8,
    "n_iter": 1,
    "latent_dim": 16,
    "n_units": 2048,
    "kl_alpha": 10000,
    "kl_beta": 0.0005,
    "batch_norm": True,
    "output_activation": "relu",
    "epochs": 20,
    "learning_rate": 0.0005,
    "batch_size": 16,
    "buffer_size": 60000,
    "buffer_size_dataset": True,
    "continue_training": False,
    "max_ckpts_to_keep": 2,
    "checkpoint_epochs": 15,
    "save_best_only": True,
    "learning_schedule": False,
    "early_patience_epoch": 50,
    "early_delta": 1e-9,
    "adam_beta_1": 0.9,
    "adam_beta_2": 0.999
}

config = wandb.config

