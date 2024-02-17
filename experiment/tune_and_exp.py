import numpy as np
from types import SimpleNamespace
from models.base import setup_model
from agents.utils.name_match import agents, agents_replay
from experiment.exp import offline_train_eval
from utils.stream import IncrementalTaskStream, get_cls_order
from utils.utils import seed_fixer, check_ram_usage, save_pickle
from utils.metrics import compute_performance, compute_performance_offline
import time
import os
from experiment.tune_config import config_generic, config_model, config_cl, set_dropout, modify_config_accordingly
from functools import partial
from ray import tune, air
import torch


def adjust_config_for_ablation(args, config_cl):
    if not args.inc:
        config_cl['inchannel_scale'] = 0
    if not args.xc:
        config_cl['xchannel_scale'] = 0
    if not args.feat:
        config_cl['feat_scale'] = 0
    if not args.freq:
        config_cl['k_freq'] = 0
    if args.linear:
        args.head = 'Linear'
    if args.aug:
        config_cl['augment_batch'] = True
    if args.no_feat_freq:
        config_cl['regularize_freq_on_feat'] = False


def tune_cl_agent_on_val_tasks(config, args, cls_order):
    Acc_across_runs = []
    for run in range(args.runs_val):
        args.run_id = run
        start = time.time()

        # Qz: Integrate tuning config into args
        run_args = SimpleNamespace(**config['generic'], **config['model'], **config['agent'], **vars(args))

        model = setup_model(run_args)
        agent = agents[run_args.agent](model=model, args=run_args)
        load_subject = True if 'Sub' in args.agent else False

        # ToDo: Remember to change back to val
        # task_stream = IncrementalTaskStream(data=args.data, scenario=args.scenario, cls_order=cls_order, split='exp')
        task_stream = IncrementalTaskStream(data=args.data, scenario=args.scenario, cls_order=cls_order, split='val')
        task_stream.setup(load_subject=load_subject)

        for task in task_stream.tasks:
            agent.learn_task(task)
            agent.evaluate(task_stream)

        end = time.time()
        print("-- Run_val {} | {} sec -- Test Acc: {} -- ".format(run, end - start,
                                                                  np.around(agent.Acc_tasks['test'][-1], decimals=2),
                                                                ))
        # Select the best params using test set
        Acc_across_runs.append(agent.Acc_tasks['test'])

        os.remove(agent.ckpt_path)  # delete the ckpt to clear the disk memory
        if args.agent == 'GR':
            os.remove(agent.ckpt_path_g)
        del model  # Clean up GPU memory
        del agent
        torch.cuda.empty_cache()

    Acc_across_runs = np.array(Acc_across_runs)
    end_avg_acc, end_avg_fgt, avg_cur_acc, _, _ = compute_performance(Acc_across_runs)

    return {'end_avg_acc': end_avg_acc[0],
            'end_avg_fgt': end_avg_fgt[0],
            'avg_cur_acc': avg_cur_acc[0],
            'Acc_across_runs': Acc_across_runs}


def tune_offline_on_val_tasks(config, args, cls_order):
    # ############### Runs Loop ###############
    Acc_across_runs = []

    for run in range(args.runs_val):
        args.run_id = run
        start = time.time()

        # Qz: Integrate tuning config into args
        run_args = SimpleNamespace(**config['generic'], **config['model'], **config['agent'], **vars(args))
        task_stream = IncrementalTaskStream(data=args.data, scenario=args.scenario, cls_order=cls_order, split='val')
        val_acc, test_acc = offline_train_eval(task_stream, run, run_args)
        end = time.time()

        print("-- Run_val {} | {} sec -- Test Acc: {} -- ".format(run, end - start, np.around(test_acc, decimals=2)))
        Acc_across_runs.append(test_acc)

    # ################## mean and CI over runs ##################
    Acc_across_runs = np.array(Acc_across_runs)
    test_offline_acc = compute_performance_offline(Acc_across_runs)
    return {'offline_acc': test_offline_acc[0]}


def tune_hyperparams_on_val_tasks(args, cls_order, config_generic={}, config_model={}, config_cl={}):
    # Modify the config according to dataset and agent
    args, config_generic, config_model, config_cl = modify_config_accordingly(args, config_generic, config_model, config_cl)

    if args.ablation:
        adjust_config_for_ablation(args, config_cl[args.agent])

    config = {'generic': config_generic,
              'model': config_model[args.encoder],
              'agent': config_cl.get(args.agent, {})}

    # Qz: Allocate the resource for each trial
    resources = {'cpu': 4, 'gpu': 1} if args.device == 'cuda' else {'cpu': 1}
    # resources = {'cpu': 8, 'gpu': 0.5} if args.device == 'cuda' else {'cpu': 1}

    if args.agent == 'Offline':
        reporter = tune.CLIReporter(metric_columns=['offline_acc'])
        trainable_with_resources = tune.with_resources(partial(tune_offline_on_val_tasks,
                                                               args=args,
                                                               cls_order=cls_order),
                                                       resources=resources)
    else:
        reporter = tune.CLIReporter(metric_columns=['end_avg_acc', 'end_avg_fgt'])
        trainable_with_resources = tune.with_resources(partial(tune_cl_agent_on_val_tasks,
                                                               args=args,
                                                               cls_order=cls_order),
                                                       resources=resources)

    tuner = tune.Tuner(trainable_with_resources,
                       param_space=config,
                       tune_config=tune.TuneConfig(
                           num_samples=1,
                           trial_name_creator=lambda t: f'{t.trial_id}',
                           chdir_to_trial_dir=True),

                       run_config=air.RunConfig(
                           name=f'{args.encoder}_{args.norm}_{args.agent}',  # Experiment name
                           local_dir=f"./result/ray_tune_results/{args.scenario}/{args.data}",
                           progress_reporter=reporter,
                           verbose=1)
                       )
    results = tuner.fit()

    if args.agent == 'Offline':
        best_trial = results.get_best_result(metric='offline_acc', mode='max', scope='last')
    else:
        best_trial = results.get_best_result(metric='end_avg_acc', mode='max', scope='last')

    print(f'Best trial config: {best_trial.config}')
    print(f'Best trial metrics: {best_trial.metrics}')

    best_params = best_trial.config
    best_params = dict(best_params['model'], **best_params['agent'], **best_params['generic'])
    return best_params


def tune_and_experiment_multiple_runs(args):
    Acc_multiple_run_valid = []
    Acc_multiple_run_test = []
    Best_params = []
    from utils.setup_elements import n_tasks_exp
    n_tasks_exp = n_tasks_exp[args.data]
    start = time.time()

    if args.agent in ['Inversion']:
        args.reuse_best = True

    # ############### Runs Loop ################
    for run in range(args.runs):
        run_start = time.time()
        args.run_id = run
        tsne_path = args.exp_path + '/tsne_r{}_'.format(run)
        seed_fixer(args.seed + run)

        # Set the class order for this run
        cls_order = get_cls_order(args.data, args.fix_order)
        print('\n ######## Run {}, cls_order :{} ########'.format(run, cls_order))

        # Tune on the val tasks
        if (args.fix_order or args.reuse_best) and run > 0:  # if class order is fixed, only tune params in the first run
            pass
        else:
            tune_args = args
            tune_args.tune = True
            tune_args.stream_split = 'val'
            best_params = tune_hyperparams_on_val_tasks(tune_args, cls_order, config_generic, config_model, config_cl)
            Best_params.append(best_params)

        # Update exp args with the best params
        exp_args = SimpleNamespace(**vars(args), **best_params)
        exp_args.tune = False
        exp_args.verbose = True  # Print the loss during training
        exp_args.stream_split = 'exp'
        print(exp_args)

        task_stream = IncrementalTaskStream(data=args.data, scenario=args.scenario, cls_order=cls_order, split='exp')

        if args.agent == 'Offline':
            val_acc, test_acc = offline_train_eval(task_stream, run, exp_args)
            # Save offline acc of this run
            Acc_multiple_run_valid.append(val_acc)
            Acc_multiple_run_test.append(test_acc)

        else:
            # Setup
            load_subject = True if 'Sub' in args.agent else False
            task_stream.setup(load_subject=load_subject)
            model = setup_model(exp_args)
            agent = agents[args.agent](model=model, args=exp_args)
            # Task Loop: Train & evaluate for each task.
            for i in range(n_tasks_exp):
                task = task_stream.tasks[i]
                agent.learn_task(task)
                agent.evaluate(task_stream, path=tsne_path)  # TSNE path

                # Plot CF matrix after finishing the final task.
                if i+1 == n_tasks_exp and args.cf_matrix:
                    cf_matrix_path = args.exp_path + '/cf{}'.format(run)
                    agent.plot_cf_matrix(path=cf_matrix_path, classes=np.arange(task_stream.n_classes))

            # Save Acc of this run
            Acc_multiple_run_valid.append(agent.Acc_tasks['valid'])
            Acc_multiple_run_test.append(agent.Acc_tasks['test'])

        run_over = time.time()
        print('\n Finish Run {}: total {} sec'.format(run, run_over - run_start))

        # del model  # Clean up GPU memory for next run
        # del agent
        torch.cuda.empty_cache()

    end = time.time()
    print('\n All runs finish. Total running time: {} sec'.format(end - start))

    # ################## Val: mean and CI over runs ##################
    print('Valid Set:')
    Acc_multiple_run_valid = np.array(Acc_multiple_run_valid)
    if args.agent == 'Offline':
        acc = compute_performance_offline(Acc_multiple_run_valid)
        print('---- Offline Accuracy with 95% CI is {} ----'.format(np.around(acc, decimals=2)))
    else:
        avg_end_acc, avg_end_fgt, avg_cur_acc, avg_acc, avg_bwtp = compute_performance(Acc_multiple_run_valid)
        print(' Avg_End_Acc {} Avg_End_Fgt {} Avg_Cur_Acc {} Avg_Acc {} Avg_Bwtp {} \n'
              .format(np.around(avg_end_acc, decimals=2), np.around(avg_end_fgt, decimals=2),
                      np.around(avg_cur_acc, decimals=2), np.around(avg_acc, decimals=2),
                      np.around(avg_bwtp, decimals=2)))

    # ################## Test: mean and CI over runs ##################
    print('Test Set:')
    Acc_multiple_run_test = np.array(Acc_multiple_run_test)

    if args.agent == 'Offline':
        acc = compute_performance_offline(Acc_multiple_run_test)
        print('---- Offline Accuracy with 95% CI is {} ----'.format(np.around(acc, decimals=2)))
    else:
        avg_end_acc, avg_end_fgt, avg_cur_acc, avg_acc, avg_bwtp = compute_performance(Acc_multiple_run_test)
        print('Avg_End_Acc {} Avg_End_Fgt {} Avg_Cur_Acc {} Avg_Acc {} Avg_Bwtp {}'
              .format(np.around(avg_end_acc, decimals=2), np.around(avg_end_fgt, decimals=2),
                      np.around(avg_cur_acc, decimals=2), np.around(avg_acc, decimals=2),
                      np.around(avg_bwtp, decimals=2)))

    # Save the results
    result = {}
    result['time'] = end - start
    result['acc_array_val'] = Acc_multiple_run_valid
    result['acc_array_test'] = Acc_multiple_run_test
    result['ram'] = check_ram_usage()
    result['best_params'] = Best_params
    save_path = args.exp_path + '/result.pkl'
    save_pickle(result, save_path)

