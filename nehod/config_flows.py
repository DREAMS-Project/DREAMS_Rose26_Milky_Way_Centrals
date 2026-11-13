
from ml_collections import config_dict

def get_config():

    config = config_dict.ConfigDict()

    config.seed = 13  # random seed
    config.workdir = 'Model/'  # where to save the model
    config.name = 'gse-prob-flows-model'   # name of the model
    config.overwrite = True  # set to False to resume training
    config.checkpoint = None   # path to model checkpoint to resume training

    # Data args
    config.data = data = config_dict.ConfigDict()
    data.data_root = 'data/'  # path to training directory
    data.data_name = 'gse_prob'  # name of the dataset in the training directory
    # output parameters of flows
    data.target_parameters = [
        'mm_time', 'gse_time', 'gse_ratio', 'ngse', 'halo_f','halo_beta',
        'halo_tot_smass','halo_sat_smass', 'halo_bmass', 'halo_bmass_frac', 'halo_sfr',
        'bulge_reff', 'bulge_mass', 'bulge_smass', 'bulge_bhmass', 'bulge_dm_frac', 'bulge_vdisp',
        'disk_vcirc', 'disk_smass', 'disk_size', 'disk_height',
        'halo_central_smass', 'halo_break', 'halo_slope_in', 'halo_slope_out', 'halo_q_in', 'halo_q_out'
    ]
    # conditioning parameters for flows
    data.conditioning_parameters = ['total_mass', "omega_m", "sigma_8", "log_sn1", "log_sn2", "log_agn1"]

    # Flows args
    # Flows hyperparameters
    config.flows = flows = config_dict.ConfigDict()
    flows.in_dim = len(data.target_parameters)
    flows.context_dim = len(data.conditioning_parameters)
    flows.hidden_dims = [32, 32, 32, 32]
    flows.projection_dims = [16, 16, 16, 16]
    flows.num_transforms = 4
    flows.dropout = 0.1

    # training and loss args
    # Training hyperparameters
    config.training = training = config_dict.ConfigDict()
    training.batch_size = 32
    training.max_steps = 50_000
    training.val_check_interval = 20
    training.log_every_n_steps = 10
    training.patience = 500

    # optimizer and scheduler args
    # gradient descent and learning rate scheduler hyperparameters
    config.optimizer = optimizer = config_dict.ConfigDict()
    optimizer.name = "AdamW"
    optimizer.lr = 5e-4
    optimizer.betas = [0.9, 0.999]
    optimizer.weight_decay = 0.01
    optimizer.grad_clip = 0.5
    config.scheduler = scheduler = config_dict.ConfigDict()
    scheduler.name = "WarmUpCosineAnnealingLR"
    scheduler.warmup_steps = 100
    scheduler.decay_steps = 1_000
    scheduler.eta_min = 1e-5
    scheduler.interval = 'step'

    return config
