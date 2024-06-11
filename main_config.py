import argparse
import sys
import os
import torch
import time
from experiment.exp import experiment_multiple_runs
from utils.utils import Logger, boolean_string


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the continual learning agent on task sequence')

    # #################### Main setting for the experiment ####################
    parser.add_argument('--agent', dest='agent', default='ER', type=str,
                        choices=['SFT', 'Offline',
                                 'LwF', 'EWC', 'SI', 'MAS', 'DT2W',
                                 'ER', 'ASER', 'DER', 'Herding', 'CLOPS', 'ER_Sub'  # ER_Sub is only applicable to DSA
                                 'Mnemonics', 'Inversion', 'GR',
                                 'FastICARL'],
                        help='Continual learning agent')

    parser.add_argument('--scenario', type=str, default='class',
                        choices=['class', 'domain'],
                        help='Scenario of the task steam. Current codes only include class-il')

    parser.add_argument('--stream_split', type=str, default='exp',
                        choices=['val', 'exp', 'all'],
                        help='The split of the tasks stream: val tasks, exp tasks or all the tasks')

    parser.add_argument('--data', dest='data', default='uwave', type=str,
                        choices=['har', 'uwave', 'dailysports', 'grabmyo', 'wisdm',
                                 'ninapro', 'sines'])

    # Backbone
    parser.add_argument('--encoder', dest='encoder', default='CNN', type=str,
                        choices=['CNN', 'TST'])

    # Classifier
    parser.add_argument('--head', dest='head', default='Linear', type=str,
                        choices=['Linear', 'CosineLinear', 'SplitCosineLinear'])
    parser.add_argument('--criterion', dest='criterion', default='CE', type=str,
                        choices=['CE', 'BCE'])  # Main classification loss and activation of head
    parser.add_argument('--ncm_classifier', dest='ncm_classifier', type=boolean_string, default=False,
                        help='Use NCM classifier or not. Only work for ER-based methods.')

    # Normalizaton layers
    parser.add_argument('--norm', dest='norm', default='BN', type=str,
                        choices=['BN', 'LN', 'IN', 'BIN', 'SwitchNorm', 'StochNorm'])

    parser.add_argument('--input_norm', dest='input_norm', default='IN', type=str,
                        choices=['LN', 'IN', 'ZScore', 'none'])  # ZScore is only applicable for Offline

    """ General params """
    parser.add_argument('--runs', dest='runs', default=1, type=int,
                        help='Number of runs')
    parser.add_argument('--epochs', dest='epochs', default=100, type=int,
                        help='Number of epochs')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', dest='lr', default=1e-3, type=float,
                        help='Learning rate')
    parser.add_argument('--lradj', type=str, default='step15')
    parser.add_argument('--early_stop', type=boolean_string, default=True)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--weight_decay', dest='weight_decay', default=0, type=float,
                        help='weight decay')
    parser.add_argument('--dropout', dest='dropout', default=0, type=float)
    parser.add_argument('--feature_dim', dest='feature_dim', type=int, default=128,
                        help='Feature dimension/d_model')
    parser.add_argument('--n_layers', dest='n_layers', type=int, default=4)

    # #################### Nuisance variables  ####################
    parser.add_argument('--tune', type=boolean_string, default=False, help='flag of tuning')
    parser.add_argument('--debug', type=boolean_string, default=True)  # save the results in a 'debug' folder
    parser.add_argument('--seed', dest='seed', default=1234, type=int)
    parser.add_argument('--device', dest='device', default='cuda', type=str)
    parser.add_argument('--verbose', type=boolean_string, default=True)
    parser.add_argument('--exp_start_time', dest='exp_start_time', type=str)
    parser.add_argument('--fix_order', type=boolean_string, default=False,
                        help='Fix the class order for different runs')
    parser.add_argument('--cf_matrix', type=boolean_string, default=True,
                        help='Plot confusion matrix or not')
    parser.add_argument('--tsne', type=boolean_string, default=False,
                        help='Visualize the feature space of learner with TSNE')
    parser.add_argument('--tsne_g', type=boolean_string, default=True,
                        help='Visualize the feature space of generator with TSNE')

    # ######################## Methods-related params ###########################
    # Experience Replay
    parser.add_argument('--er_mode', type=str, default='task', choices=['online', 'task'],
                        help='Collect mem samples online or after_task')
    parser.add_argument('--mem_budget', type=float, default=0.05, help='Percentage of mem_budget/# train data')
    parser.add_argument('--buffer_tracker', type=boolean_string, default=False)  # Never be ture
    parser.add_argument('--der_plus', type=boolean_string, default=False)  # DER++
    parser.add_argument('--er_sub_type', type=str, default='part', choices=['balanced', 'part'],
                        help='2 variant of ER to utilize subject labels.')

    # KD: LwF / DT2W
    parser.add_argument('--teacher_eval', type=boolean_string, default=False)  # As Online CL survey and Avalanche
    parser.add_argument('--lambda_kd_lwf', dest='lambda_kd_lwf', default=1, type=float)
    parser.add_argument('--lambda_kd_fmap', dest='lambda_kd_fmap', default=1e-2, type=float,
                        help='lambda for KD loss on feature map')
    parser.add_argument('--fmap_kd_metric', dest='fmap_kd_metric', default="dtw", type=str,
                        choices=['dtw', 'euclidean', 'pod_temporal', 'pod_variate'],
                        help='KD metric for temporal feature map')
    parser.add_argument('--lambda_protoAug', dest='lambda_protoAug', default=100, type=float,
                        help='lambda for protoAug, no implementing this technique if lambda=0')
    parser.add_argument('--adaptive_weight',  type=boolean_string, default=False,
                        help='use linear adaptive lambda or not')

    # EWC / MAS / SI
    parser.add_argument('--lambda_impt', dest='lambda_impt', default=10000, type=float)
    parser.add_argument('--ewc_mode', dest='ewc_mode', default="separate", type=str,
                        choices=['separate', 'online'],
                        help='Mode for EWC, "separate" or "online"')

    # ASER
    parser.add_argument('--aser_k', dest='aser_k', default=3,
                        type=int,
                        help='Number of nearest neighbors (K) to perform ASER (default: %(default)s)')

    parser.add_argument('--aser_type', dest='aser_type', default="asvm", type=str, choices=['neg_sv', 'asv', 'asvm'],
                        help='Type of ASER: '
                             '"neg_sv" - Use negative SV only,'
                             ' "asv" - Use extremal values of Adversarial SV and Cooperative SV,'
                             ' "asvm" - Use mean values of Adversarial SV and Cooperative SV')

    parser.add_argument('--aser_n_smp_cls', dest='aser_n_smp_cls', default=4,
                        type=float,
                        help='Maximum number of samples per class for random sampling (default: %(default)s)')

    # CLOPS
    parser.add_argument('--mc_retrieve',  type=boolean_string, default=False,
                        help='use mc dropout retrieve strategy or not')
    parser.add_argument('--beta_lr', dest='beta_lr', default=1e-4, type=float)
    parser.add_argument('--lambda_beta', dest='lambda_beta', default=1, type=float)

    # Generative Replay
    parser.add_argument('--epochs_g', type=float, default=500)
    parser.add_argument('--lr_g', type=float, default=1e-3)
    parser.add_argument('--recon_wt', type=float, default=0.1)

    # Mnemonics
    parser.add_argument('--mnemonics_epochs', default=1, type=int)
    parser.add_argument('--mnemonics_lr', type=float, default=1e-5)

    # Model Inversion
    parser.add_argument('--start_noise', default=True, type=boolean_string)
    parser.add_argument('--save_mode', default=0, type=int, choices=[0, 1, 2, 3])
    parser.add_argument('--n_samples_to_plot', default=5, type=int)
    parser.add_argument('--augment_batch', default=False, type=boolean_string)
    parser.add_argument('--visual_syn_feat', default=True, type=boolean_string)
    parser.add_argument('--iterations_per_layer', type=int, default=100)
    parser.add_argument('--inversion_lr', type=float, default=1e-2)
    parser.add_argument('--inchannel_scale', type=float, default=10)
    parser.add_argument('--xchannel_scale', type=float, default=1)
    parser.add_argument('--feat_scale', type=float, default=1)  # No loss_feat if == 0
    parser.add_argument('--k_freq', type=int, default=-1)  # All freq if == -1, no loss_freq if == 0
    parser.add_argument('--regularize_freq_on_feat', default=False, type=boolean_string)

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set directories
    exp_start_time = time.strftime("%b-%d-%H-%M-%S", time.localtime())
    exp_path_0 = './result/exp/' if not args.debug else './result/exp/debug'
    exp_path_1 = args.encoder + '_' + args.data
    exp_path_2 = args.agent + '_' + args.norm + '_' + exp_start_time
    exp_path = os.path.join(exp_path_0, exp_path_1, exp_path_2)
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    log_dir = exp_path + '/' + 'log.txt'
    sys.stdout = Logger(log_dir)
    args.exp_path = exp_path  # One experiment with multiple runs
    print(args)

    experiment_multiple_runs(args)