"""
Hyperparameter search-space configuration.
"""
from ray import tune
import os
from agents.utils.name_match import agents_replay

content_root = os.path.abspath('.')
config_default = {'scenario': 'class',
                  'runs': 5,
                  'runs_val': 2,  # To increase the confidence of best_config
                  'seed': 1234,
                  'verbose': False,
                  'path_prefix': content_root + '/result/tune_and_exp',
                  'fix_order': False,
                  'reuse_best': False,
                  'tsne': False,
                  'tsne_g': False,
                  'cf_matrix': True,
                  'epochs': 100,  # Early stop in most cases.
                  'tune': False,
                  'ablation': False,
                  'teacher_eval': False,
                  'er_mode': 'task',
                  'buffer_tracker': False,
                  'early_stop': True,
                  'patience': 5,
                  'input_norm': 'LN',
                  'mem_budget': 0.05,  # 0.01 0.05 (default)  0.1  0.2  1
                  'head': 'Linear',  # Linear (Default),  SplitCosineLinear
                  'criterion': 'CE',  # Default: CE (Default),  BCE
                  'ncm_classifier': False,  # Default: False
                  'er_sub_type': 'balanced'    # 'part' / 'balanced'
                  }
                  # 1. Linear, BCE, False; 2. Linear, CE, True; 3. SplitCosineLinear, CE, False


# Hyperparameter search-space
config_generic = {'lr': 1e-3,  # Tried with 1e-2 & 1e-4, never be the best. So we focus on lradj.
                  'lradj': tune.grid_search(['TST', 'step10', 'step15']),  # 'step25' seldom be the best, removed.
                  'batch_size': 64,  # 64 is mostly better than 32. So we decide directly set it as 64.
                  'weight_decay': 0,  # Trivial, also found harmful for CL in recent study
                  }


config_model = {'CNN': {'feature_dim': 128,
                        'n_layers': 4,
                        'dropout': 0.3,  # Select an appropriate dropout value for each the dataset
                        },

                'TST': {'feature_dim': 128,
                        'n_layers': 3,
                        'dropout': 0.3,  # Select an appropriate dropout value for each the dataset
                        }
                }


config_cl = {'LwF': {'lambda_kd_lwf': tune.grid_search([1, 1e1, 1e2, 1e3, 1e4]),
                     'adaptive_weight': True,
                     },

             'EWC': {'lambda_impt': tune.grid_search([1e2, 1e3, 1e4]),
                     'ewc_mode': 'separate'},

             'SI': {'lambda_impt': tune.grid_search([1e3, 1e4, 1e5])},

             'MAS': {'lambda_impt': tune.grid_search([1e2, 1e3, 1e4])},

             'DT2W': {'lambda_kd_lwf': tune.grid_search([1, 10]),
                      'lambda_kd_fmap': tune.grid_search([1e-1, 1e-2, 1e-3]),
                      'lambda_protoAug': tune.grid_search([10, 100]),
                      'adaptive_weight': True,
                      'fmap_kd_metric': 'dtw'},

             'DER': {'der_plus': False},

             'CLOPS': {'mc_retrieve': False,  # mc_retrieve is highly time-consuming. Yet the results are quite poor.
                       'beta_lr': tune.grid_search([1e-4, 1e-5]),
                       'lambda_beta': tune.grid_search([0.1, 1, 10, 100])
                       },

             'ASER': {'aser_k': tune.grid_search([3]),
                      'aser_type': 'asvm',
                      'aser_n_smp_cls': tune.grid_search([2, 4])
                      },

             'GR': {'lr_g': tune.grid_search([1e-3, 1e-4]),
                    'epochs_g': 500,  # Using earlystop with patience = 20
                    'recon_wt': tune.grid_search([1e-3, 0.01, 0.1, 1, 10]),
                    'adaptive_weight': False},  # True

             'Mnemonics': {'mnemonics_epochs': tune.grid_search([5, 10, 50]),
                           'mnemonics_lr': tune.grid_search([1e-3, 1e-4]),
                           },

             'Inversion': {'inchannel_scale': tune.grid_search([10, 1]),
                           'xchannel_scale': tune.grid_search([1, 0.1]),
                           'feat_scale': tune.grid_search([10, 1, 0.1]),
                           'iterations_per_layer': tune.grid_search([500]),
                           'inversion_lr': tune.grid_search([1e-2, 1e-3]),
                           'start_noise': True,
                           'save_mode': 1,
                           'n_samples_to_plot': 0,
                           'visual_syn_feat': False,
                           'k_freq': 0,  # -1
                           'augment_batch': False,
                           'regularize_freq_on_feat': True
                           }
             }


def set_dropout(data):
    if data == 'dailysports':
        dropout = 0.3
    elif 'grabmyo' in data:
        dropout = 0.3
    elif data == 'har':
        dropout = 0
    elif data == 'uwave':
        dropout = 0
    elif data == 'wisdm':
        dropout = 0.3
    else:
        raise ValueError("No such dataset")
    return dropout


def modify_config_accordingly(args, config_generic, config_model, config_cl):
    """
    Modify the config according to dataset and agent.
    It helps to reduce the size of params grid for tuning.
    """
    # Half the batch_size for relay-based methods
    # Reasons: 1. reduce GPU usage; 2. buffer size for some datasets is smaller than 64.
    if args.agent in agents_replay:
        config_generic['batch_size'] = 32

    if args.agent in agents_replay or args.agent == 'Offline':  # Set a larger patience for methods using replay
        args.patience = 20

    # Set dropout based on the dataset
    config_model[args.encoder]['dropout'] = set_dropout(args.data)

    # HAR
    if args.data == 'har':
        if args.agent == 'DT2W':
            config_cl[args.agent]['lambda_kd_fmap'] = tune.grid_search([1e-1, 1e-2])  # 1e-3 never be the best

    # UWave
    if args.data == 'uwave':
        args.input_norm = 'IN'
        if args.agent == 'DT2W':
            config_cl[args.agent]['lambda_kd_fmap'] = tune.grid_search([1e-2, 3e-3, 1e-3])

    # Dailysports
    if args.data == 'dailysports':
        if args.agent == 'LwF':
            config_cl[args.agent]['adaptive_weight'] = tune.grid_search([False])

        if args.agent == 'DT2W':
            config_cl[args.agent]['adaptive_weight'] = tune.grid_search([False])

    # GRABMyo
    if 'grabmyo' in args.data:
        if args.agent == 'ASER':
            config_cl[args.agent]['aser_n_smp_cls'] = tune.grid_search([4, 8, 12])  # Tried with 2, never be the best

        if args.agent == 'MAS':
            config_cl[args.agent]['lambda_impt'] = tune.grid_search([1, 10, 1e2, 1e3])  # 1e4 never be the best

    if args.agent in ['Inversion', 'Mnemonics']:
        config_generic['lradj'] = 'TST'  # experiments showed 'TST' is always the best. Set it directly to reduce grid.

    if args.agent == 'Inversion':
        args.reuse_best = True
        args.runs_val = 1

    # WISDM
    if args.data == 'wisdm':
        args.input_norm = 'none'
        if args.agent == 'ASER':
            config_cl[args.agent]['aser_k'] = tune.grid_search([3, 5])
            config_cl[args.agent]['aser_n_smp_cls'] = tune.grid_search([2, 4, 8, 12])  # Tried with 2, never be the best

    if args.agent == 'GR':
        args.runs_val = 1  # Hyper params is trivial for GR's performance. Set to 1 to speed up.

    return args, config_generic, config_model, config_cl
