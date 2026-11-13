
import os
import shutil
import sys

import ml_collections
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import torch
import yaml
from absl import flags, logging
from ml_collections import config_flags

import datasets
from nehod import flows, utils
import optuna
import numpy as np
import matplotlib.pyplot as plt

logging.set_verbosity(logging.INFO)

def read_flows_dataset(config):
    """ Hacky way to read flows data using datasets. Not optimized but works. """

    parameters = config.data.target_parameters + config.data.conditioning_parameters
    num_target = len(config.data.target_parameters)

    # Read and preprocess the dataset
    _, train_params, _, norm_dict = datasets.read_preprocess_dataset(
        data_root=config.data.data_root,
        data_name=config.data.data_name,
        conditioning_parameters=parameters,
        flag='train',
    )
    _, val_params, _, _ = datasets.read_preprocess_dataset(
        data_root=config.data.data_root,
        data_name=config.data.data_name,
        conditioning_parameters=parameters,
        flag='val',
        norm_dict=norm_dict,
    )

    # Separate the target and conditioning parameters
    train_target = train_params[:, :num_target]
    train_cond = train_params[:, num_target:]
    val_target = val_params[:, :num_target]
    val_cond = val_params[:, num_target:]

    # Create the dataloaders
    train_loader = datasets.create_dataloader(
        (train_target, train_cond), batch_size=config.training.batch_size,
        shuffle=True, pin_memory=torch.cuda.is_available())
    val_loader = datasets.create_dataloader(
        (val_target, val_cond), batch_size=config.training.batch_size,
        shuffle=False, pin_memory=torch.cuda.is_available())

    # Restructure norm_dict
    new_norm_dict = {
        'target_mean': norm_dict['cond_mean'][:num_target],
        'target_std': norm_dict['cond_std'][:num_target],
        'cond_mean': norm_dict['cond_mean'][num_target:],
        'cond_std': norm_dict['cond_std'][num_target:],
    }

    return train_loader, val_loader, new_norm_dict

def train(trial,
    config: ml_collections.ConfigDict, workdir: str = "./logging/"
):

    # Get new hyperparameters
    learning_rate = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float("wd", 1e-4, 5e-1, log=True)
    dropout = trial.suggest_float("dr", 1e-4, 5e-1, log=True)
    num_transforms = trial.suggest_int("nt", 1, 10)
    print("HPARAMS", learning_rate, weight_decay, dropout, num_transforms)

    #learning_rate = 0.0008089985010158357
    #weight_decay = 0.05822553145128255
    #dropout = 0.0386717577803789
    #num_transforms = 8

    # set up work directory
    if not hasattr(config, "name"):
        name = utils.get_random_name()
    else:
        name = config["name"]
    name += f'_{trial.number}'
    logging.info("Starting training run {} at {}".format(name, workdir))

    # set up random seed
    pl.seed_everything(config.seed)

    workdir = os.path.join(workdir, name)
    checkpoint_path = None
    if os.path.exists(workdir):
        if config.overwrite:
            shutil.rmtree(workdir)
        elif config.get('checkpoint', None) is not None:
            checkpoint_path = os.path.join(
                workdir, 'lightning_logs/checkpoints', config.checkpoint)
        else:
            raise ValueError(
                f"Workdir {workdir} already exists. Please set overwrite=True "
                "to overwrite the existing directory.")

    os.makedirs(workdir, exist_ok=True)

    # Read in and prepare the dataset
    train_loader, val_loader, norm_dict = read_flows_dataset(config)

    # Apply this trial's hyperparameters. Release fix: the research code evaluated
    # `config.flows.num_transforms` without assigning it, so every model used in the paper
    # has the config default depth of 4 (see ISSUES.md).
    config.flows.num_transforms = num_transforms
    config.flows.dropout = dropout
    config.optimizer.lr = learning_rate
    config.optimizer.weight_decay = weight_decay

    # Save the configuration (written after the trial values are applied, so it records them)
    config_dict = config.to_dict()
    config_path = os.path.join(workdir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    # Create the flows model
    model = flows.NPE(
        in_dim=config.flows.in_dim,
        context_dim=config.flows.context_dim,
        hidden_dims=config.flows.hidden_dims,
        projection_dims=config.flows.projection_dims,
        dropout=config.flows.dropout,
        num_transforms=config.flows.num_transforms,
        optimizer_args=config.optimizer,
        scheduler_args=config.scheduler,
        norm_dict=norm_dict,
    )

    # Create Trainer and start training
    callbacks = [
        pl.callbacks.EarlyStopping(
            monitor='val_loss', patience=config.training.patience,
            mode='min', verbose=True),
        pl.callbacks.ModelCheckpoint(
            filename="{epoch}-{val_loss:.4f}", monitor='val_loss',
            save_last=True, save_top_k=10, mode='min', save_weights_only=False),
        pl.callbacks.LearningRateMonitor("step"),
    ]
    #train_logger = pl_loggers.TensorBoardLogger(workdir, version='')
    train_logger = pl_loggers.CSVLogger(workdir, version='')

    trainer = pl.Trainer(
        default_root_dir=workdir,
        max_steps=config.training.max_steps,
        accelerator='gpu',
        callbacks=callbacks,
        logger=train_logger,
        enable_progress_bar=False,
        gradient_clip_val=config.optimizer.grad_clip,
        val_check_interval=config.training.val_check_interval,
        check_val_every_n_epoch=None,
        log_every_n_steps=config.training.log_every_n_steps,
    )

    # train the model
    trainer.fit(
        model, train_loader, val_loader,
        ckpt_path=checkpoint_path
    )


    log_file = f'Model/{name}/lightning_logs/metrics.csv'
    with open(log_file, 'r') as ofile:
        loss_cols = ofile.readline().strip().split(',')

    epoch_idx = train_idx = val_idx = None
    for i,col in enumerate(loss_cols):
        if col == 'epoch':
            epoch_idx = i
        if col == 'train_loss_epoch':
            train_idx = i
        if col == 'val_loss_epoch':
            val_idx = i

    if epoch_idx is None or train_idx is None or val_idx is None:
        print("Log File Read Failed!!")
        return 1e3

    loss_data = np.genfromtxt(log_file, delimiter=',') 

    tcut = np.isnan(loss_data[:,train_idx]) #0
    train = loss_data[~tcut,train_idx] #0
    train_epoch = loss_data[~tcut,epoch_idx] #1

    vcut = np.isnan(loss_data[:,val_idx]) #5
    valid = loss_data[~vcut,val_idx] #5
    valid_epoch = loss_data[~vcut, epoch_idx]

    fig, ax = plt.subplots()
    ax.plot(train_epoch, train, label='Train')
    ax.plot(valid_epoch, valid, label='Validation')
    ax.legend(frameon=False)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    fig.savefig(f"plots/loss/flow_loss_{name}.pdf", bbox_inches='tight')
    plt.close(fig)

    ncut = ~np.isnan(valid)
    idx = np.argmin(valid[ncut])

    min_loss = np.average(valid[ncut][idx-25:idx+25])
    print('min_loss', min_loss)

    return min_loss



if __name__ == "__main__":
    FLAGS = flags.FLAGS
    config_flags.DEFINE_config_file(
        "config",
        "./config_flows.py",
        lock_config=True,
    )
    # Parse flags
    FLAGS(sys.argv)

    # Optuna parameters
    storage = "sqlite:///"+os.getcwd()+"/databases/optuna_master_cdm"
    n_trials   = 50

    sampler = optuna.samplers.TPESampler(n_startup_trials=20)
    study = optuna.create_study(study_name='master_cdm_flow', sampler=sampler, storage=storage, load_if_exists=True)

    def objective(trial):
        min_loss = train(trial, config=FLAGS.config, workdir=FLAGS.config.workdir)
        return min_loss

    study.optimize(objective, n_trials, gc_after_trial=True)

    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Start training run
    #train(config=FLAGS.config, workdir=FLAGS.config.workdir)
