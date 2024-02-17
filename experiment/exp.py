# -*- coding: UTF-8 -*-
import torch
import time
import os
import numpy as np
from agents.utils.name_match import agents
from agents.utils.functions import epoch_run, test_epoch_for_cf_matrix
from models.base import setup_model
from utils.stream import IncrementalTaskStream, get_cls_order
from utils.metrics import compute_performance, compute_performance_offline
from utils.data import Dataloader_from_numpy
from utils.optimizer import set_optimizer, adjust_learning_rate
from utils.utils import seed_fixer, EarlyStopping, check_ram_usage, save_pickle, BinaryCrossEntropy
from utils.metrics import plot_confusion_matrix
from torch.optim import lr_scheduler


def offline_train_eval(task_stream, run, args):
    assert args.head != 'SplitCosineLinear'

    # Combine all the data in the task stream
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = task_stream.setup_offline()

    if args.input_norm == 'ZScore':
        mean = np.mean(x_train, axis=(0, 1))
        std = np.std(x_train, axis=(0, 1))
        x_train = (x_train - mean) / std
        x_val = (x_val - mean) /std
        x_test = (x_test - mean) / std

    train_loader = Dataloader_from_numpy(x_train, y_train.squeeze(), batch_size=args.batch_size, shuffle=True)
    del x_train, y_train
    val_loader = Dataloader_from_numpy(x_val, y_val.squeeze(), batch_size=args.batch_size, shuffle=False)
    del x_val, y_val
    test_loader = Dataloader_from_numpy(x_test, y_test.squeeze(), batch_size=args.batch_size, shuffle=False)
    del x_test, y_test

    assert args.head != 'SplitCosineLinear'
    # Initialization
    model = setup_model(args)
    opt = set_optimizer(model, args)

    # set classification criterion
    if args.criterion == 'BCE':
        criterion = BinaryCrossEntropy(dim=model.head.out_features, device=args.device)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # set ckpt path when tuning
    if args.tune:
        ckpt_path = args.exp_path + '/ckpt_{}_r{}.pt'.format(os.getpid(), run)
    else:
        ckpt_path = args.exp_path + '/ckpt_r{}.pt'.format(run)

    early_stopping = EarlyStopping(path=ckpt_path, patience=args.patience, mode='min', verbose=False)

    scheduler = lr_scheduler.OneCycleLR(optimizer=opt,
                                        steps_per_epoch=len(train_loader),
                                        epochs=args.epochs,
                                        max_lr=args.lr)

    # Train
    for epoch in range(args.epochs):
        epoch_loss, epoch_acc = epoch_run(model, train_loader, opt, scheduler, criterion, epoch, args, train=True)
        epoch_loss_val, epoch_acc_val = epoch_run(model, val_loader, opt, scheduler, criterion, epoch, args, train=False)

        if args.lradj != 'TST':
            adjust_learning_rate(opt, scheduler, epoch + 1, args)

        early_stopping(epoch_loss_val, model)
        if early_stopping.early_stop:
            if args.verbose:
                print("Early stopping")
            break

        if args.verbose:
            print('Epoch {}: training loss: {}, training accuracy: {}'.format(epoch, epoch_loss, epoch_acc))

    # Test:
    model.load_state_dict(torch.load(ckpt_path))
    if args.tune:
        os.remove(ckpt_path)  # delete the ckpt to free up disk memory

    # Eval on val set
    val_loss, val_acc = epoch_run(model, val_loader, opt, scheduler, criterion, None, args, train=False)

    # Eval on test set
    if args.cf_matrix and args.tune is False:
        test_loss, test_acc, y_pred, y_true = test_epoch_for_cf_matrix(model, test_loader, criterion, device=args.device)
        cf_matrix_path = args.exp_path + '/cf{}'.format(run)
        plot_confusion_matrix(y_true, y_pred, path=cf_matrix_path, classes=np.arange(task_stream.n_classes))
    else:
        test_loss, test_acc = epoch_run(model, test_loader, opt, scheduler, criterion, None, args, train=False)

    if args.verbose:
        print('Run {} val loss {}, val accuracy: {:.2f} ; test loss {}, test accuracy: {:.2f}'.format(run, val_loss, val_acc,
                                                                                                      test_loss, test_acc))

    return val_acc, test_acc


def experiment_multiple_runs(args):
    """
    Multiple runs for single head model on dataset

    :return:
    """

    # ############### Runs Loop ################
    start = time.time()
    Acc_multiple_run_valid = []
    Acc_multiple_run_test = []

    for run in range(args.runs):
        args.run_id = run
        tsne_path = args.exp_path + '/tsne_r{}_'.format(run)
        run_start = time.time()

        # Fix with different seed for each run。
        seed_fixer(args.seed + run)
        cls_order = get_cls_order(args.data, args.fix_order)
        print('\n ######## {} tasks, Run {}, cls_order :{} ########'.format(args.stream_split, run, cls_order))
        task_stream = IncrementalTaskStream(data=args.data, scenario=args.scenario, cls_order=cls_order, split=args.stream_split)

        if args.agent == 'Offline':
            val_acc, test_acc = offline_train_eval(task_stream, run, args)
            Acc_multiple_run_valid.append(val_acc)
            Acc_multiple_run_test.append(test_acc)
        else:
            load_subject = True if 'Sub' in args.agent else False
            task_stream.setup(load_subject=load_subject)
            model = setup_model(args)
            agent = agents[args.agent](model=model, args=args)
            # Task Loop: Train & evaluate for each task. Plot CF matrix after finishing the final task.
            for i in range(task_stream.n_tasks):
                task = task_stream.tasks[i]
                agent.learn_task(task)
                agent.evaluate(task_stream, path=tsne_path)  # TSNE path

                # Plot CF matrix after finishing the final task.
                if i + 1 == task_stream.n_tasks and args.cf_matrix:
                    cf_matrix_path = args.exp_path + '/cf{}'.format(run)
                    agent.plot_cf_matrix(path=cf_matrix_path, classes=np.arange(task_stream.n_classes))
            Acc_multiple_run_valid.append(agent.Acc_tasks['valid'])
            Acc_multiple_run_test.append(agent.Acc_tasks['test'])

        run_over = time.time()
        print('\n Finish Run {}, running time {} sec'.format(run, run_over - run_start))
    end = time.time()
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
    save_path = args.exp_path + '/result.pkl'
    save_pickle(result, save_path)

