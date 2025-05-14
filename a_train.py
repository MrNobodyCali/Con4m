import os
import sys
import time
import json
import torch
import argparse
import numpy as np
from copy import deepcopy

sys.path.append('utils')
sys.path.append('pipeline')
sys.path.append('model')

import utils.misc as utils
from utils.default_config import set_default_config, get_choice_default_config
from pipeline.c_dataset_dataloader import CLDataSet
from model.consistent_predict_encoder import CLModel
from model.consistent_label_transformation import CLTransform

num_threads = '32'
torch.set_num_threads(int(num_threads))
os.environ['OMP_NUM_THREADS'] = num_threads
os.environ['OPENBLAS_NUM_THREADS'] = num_threads
os.environ['MKL_NUM_THREADS'] = num_threads
os.environ['VECLIB_MAXIMUM_THREADS'] = num_threads
os.environ['NUMEXPR_NUM_THREADS'] = num_threads


def train_step(
        dataloader,
        model,
        transform,
        optimizer,
        update_label,
        epoch_num,
        logging_step,
):
    model.train()
    device = next(model.parameters()).device

    start_time = time.perf_counter()
    n_examples = 0
    logs, last_logs = {}, None
    iter_count = 0
    batch_tot_loss = 0

    for step, full_data in enumerate(dataloader):
        batch_data, batch_label, eta, index = full_data
        n_examples += batch_data.size(0)
        batch_data = batch_data.to(device)
        batch_label = batch_label.to(device)
        eta = eta.to(device)

        loss, hat_p, tilde_p, seg_y = \
            model(
                x=batch_data,
                y=batch_label,
            )
        # Step 4: obtain the consistent labels
        c_pred = transform.process_batch_label(
            hat_p,
            tilde_p,
            index,
            eta,
            update_label,
            epoch_num,
        )
        acc = c_pred.eq(seg_y).float().mean()

        last_logs = utils.batch_logs_update('train', logs, last_logs, loss, acc, step, logging_step,
                                            start_time, n_examples)
        loss.backward()
        batch_tot_loss += loss.detach().cpu().numpy()

        if n_examples % 64 == 0:
            optimizer.step()
            optimizer.zero_grad()
        iter_count += 1

    optimizer.step()
    optimizer.zero_grad()

    logs = utils.update_logs(logs, iter_count)
    logs["train_iter"] = iter_count
    utils.show_logs("Average training loss on epoch", logs)

    return logs, batch_tot_loss


def valid_step(
        dataloader,
        model,
        transform,
        logging_step,
):
    model.eval()
    device = next(model.parameters()).device
    start_time = time.perf_counter()
    n_examples = 0

    logs = {}
    iter_count = 0
    batch_tot_loss = 0

    true_label = torch.tensor([], dtype=torch.long)
    pred_label = torch.tensor([], dtype=torch.long)

    for step, full_data in enumerate(dataloader):
        batch_data, batch_label, _, _ = full_data
        n_examples += batch_data.size(0)
        batch_data = batch_data.to(device)
        batch_label = batch_label.to(device)

        with torch.no_grad():
            loss, hat_p, tilde_p, seg_y = \
                model(
                    x=batch_data,
                    y=batch_label,
                )
        # Step 4: obtain the consistent labels
        c_pred = transform.process_batch_label(
            hat_p,
            tilde_p,
            index=None,
            eta=None,
            update_label=False,
            epoch_num=None,
        )
        acc = c_pred.eq(seg_y).float().mean()

        logs = utils.batch_logs_update('valid', logs, None, loss, acc)
        true_label = torch.cat((true_label, torch.argmax(batch_label, dim=-1).view(-1).cpu()))
        pred_label = torch.cat((pred_label, c_pred.view(-1).cpu()))

        batch_tot_loss += loss.detach().cpu().numpy()
        iter_count += 1

        if (step + 1) % logging_step == 0:
            new_time = time.perf_counter()
            elapsed = new_time - start_time
            print(f"Update {step + 1}")
            print(f"elapsed: {elapsed:.1f} s")
            print(f"{1000.0 * elapsed / (logging_step * (step + 1)):.1f} ms per batch, "
                  f"{1000.0 * elapsed / n_examples:.1f} ms / example")
            print('-' * 50)

    logs = utils.update_logs(logs, iter_count)
    logs["valid_iter"] = iter_count
    utils.show_logs("Average validation loss:", logs)

    index = dataloader.dataset.data_handler.model_evaluation(
        true_label.numpy(),
        pred_label.numpy(),
        dataloader.dataset.n_class,
    )
    print('-' * 10, 'The average validation results', '-' * 10)
    print(index)

    return logs, batch_tot_loss, index.f1


def change_step(
        ori_label,
        correct_label,
):
    ori_label = torch.argmax(ori_label, dim=-1).view(-1)
    correct_label = torch.argmax(correct_label, dim=-1).view(-1)

    # the total number of labels that were modified by the model during training
    total_change_num = ori_label.ne(correct_label).sum()
    print(f'The number of changed labels from original labels is: {total_change_num}')

    return total_change_num


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLModel')
    parser = set_default_config(parser)

    group_train = parser.add_argument_group('Train')
    group_train.add_argument('--database_save_dir', type=str, default='/data/eeggroup/CL_database/',
                             help='Should give a path to load the database.')
    group_train.add_argument('--data_name', type=str, default='fNIRS_2',
                             help='Should give the name of the database [fNIRS_2, Sleep, HHAR].')
    group_train.add_argument('--noise_ratio', type=float, default=.0,
                             help='The noisy ratio of the loading dataset.')
    group_train.add_argument('--exp_id', type=int, default=1,
                             help='The experimental id. fNIRS_2: 1-12; Sleep/HHAR: 1-6.')
    group_train.add_argument('--gpu_id', type=int, default=0,
                             help='The gpu id.')
    group_train.add_argument('--batch_size', type=int, default=64,
                             help='Number of batches.')
    group_train.add_argument('--save_step', type=int, default=10,
                             help='The step number to save checkpoint.')
    group_train.add_argument('--all_epoch_num', type=int, default=100,
                             help='Epoch number for total iterations in all stages.')
    group_train.add_argument('--loss_change', type=float, default=1e-3,
                             help='The convergence tolerance value to stop training.')
    group_train.add_argument('--early_stop', action='store_false',
                             help='Whether to use early stopping technique during training.')
    group_train.add_argument('--patience', type=int, default=10,
                             help='The waiting epoch number for early stopping.')
    group_train.add_argument('--load_path', type=str, default=None,
                             help='The path to load checkpoint.')
    group_train.add_argument('--load_best', action='store_false',
                             help='Whether to load the best state in the checkpoint.')
    group_train.add_argument('--best_val_index', type=str, default='F1',
                             help='The index for saving models performing best in the validation dataset. The candidate'
                                  'list includes: [loss, F1].')
    group_train.add_argument('--path_checkpoint', type=str, default='/data/eeggroup/CL_result/',
                             help='The path to save checkpoint.')
    group_train.add_argument('--lr', type=float, default=1e-3,
                             help='The learning rate.')
    group_train.add_argument('--weight_decay', type=float, default=1e-4,
                             help='The weight decay.')
    group_train.add_argument('--gpu', action='store_false',
                             help='Whether to load the data and model to GPU.')

    # Load the config
    argv = sys.argv[1:]
    args = parser.parse_args(argv)
    args, config = get_choice_default_config(args)
    main_logs = {"epoch": []}

    # Load the datasets
    args.patient_list = args.train_patient_list
    train_dataset = CLDataSet(args)
    args.patient_list = args.valid_patient_list
    valid_dataset = CLDataSet(args)
    valid_dataset.epoch_num = args.all_epoch_num

    config.n_class = train_dataset.n_class
    config.seg_small_num = train_dataset.seg_small_num
    config.raw_input_len = train_dataset.data_handler.window_len
    config.n_features = train_dataset.n_features

    # Construct the model
    cl_model = CLModel(config)
    ori_y = train_dataset.get_initial_label()
    cl_transform = CLTransform(
        ori_y=ori_y,
        batch_size=args.batch_size,
        seg_num=config.seg_small_num,
    )

    # Optimizer
    cl_optimizer = torch.optim.Adam(
        cl_model.parameters(),
        args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.weight_decay,
    )

    best_valid_loss = np.inf
    best_valid_f1 = -1
    last_valid_loss = [np.inf for _ in range(args.patience)]
    last_valid_f1 = [-1 for _ in range(args.patience)]
    assert args.best_val_index in ['loss', 'F1']
    if args.load_path is not None:
        args.load_path = os.path.join(args.load_path, f'{args.data_name}/Con4m/')
        args.load_path = os.path.join(args.load_path, f'{int(args.noise_ratio * 100)}/')
        args.path_checkpoint = os.path.join(args.path_checkpoint, 'exp' + str(args.exp_id))

        load_path, main_logs = utils.load_checkpoint(args.load_path)
        print('-' * 50)
        print('Load checkpoint:', load_path)

        state_dict = torch.load(load_path, 'cpu')
        if args.load_best:
            if args.best_val_index == 'loss':
                cl_model.load_state_dict(state_dict["BestLossModel"], strict=False)
            else:
                cl_model.load_state_dict(state_dict["BestF1Model"], strict=False)
        else:
            cl_model.load_state_dict(state_dict["CLModel"], strict=False)
        best_valid_loss = state_dict["BestValLoss"]
        best_valid_f1 = state_dict["BestValF1"]
        best_loss_model_state = state_dict["BestLossModel"]
        best_f1_model_state = state_dict["BestF1Model"]
        last_model_state = state_dict["CLModel"]
    else:
        best_loss_model_state = deepcopy(cl_model.state_dict())
        best_f1_model_state = deepcopy(cl_model.state_dict())
        last_model_state = deepcopy(cl_model.state_dict())

    if args.gpu:
        gpu_device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
        cl_model.to(gpu_device)
        cl_transform.l_tanh.to(gpu_device)

    if args.load_path is not None:
        cl_optimizer.load_state_dict(state_dict["Optimizer"])

    path_checkpoint = None
    if args.path_checkpoint is not None:
        args.path_checkpoint = os.path.join(args.path_checkpoint, f'{args.data_name}/Con4m/')
        args.path_checkpoint = os.path.join(args.path_checkpoint, f'{int(args.noise_ratio * 100)}/')
        args.path_checkpoint = os.path.join(args.path_checkpoint, 'exp' + str(args.exp_id))

        if not os.path.exists(args.path_checkpoint):
            os.makedirs(args.path_checkpoint)
        path_checkpoint = os.path.join(args.path_checkpoint, "checkpoint")

    print(f'CONFIG:\n{json.dumps(vars(args), indent=4, sort_keys=True)}')
    print('-' * 50)

    print(f"Running {args.all_epoch_num} epochs")
    start_epoch = len(main_logs["epoch"])
    last_loss = 0
    max_total_change_num = 0
    wait_epoch = 0

    main_start_time = time.time()
    for epoch in range(start_epoch, args.all_epoch_num):
        print('-' * 50)
        if epoch < args.warm_epoch_num:
            warm_up_stage = True
            print(f"Starting warm-up epoch {epoch}")
        else:
            warm_up_stage = False
            print(f"Starting CL epoch {epoch}")
        utils.cpu_stats()

        train_loader = train_dataset.get_data_loader(args.batch_size, shuffle=True, num_workers=0)
        valid_loader = valid_dataset.get_data_loader(args.batch_size, shuffle=False, num_workers=0)

        print("Training dataset %d batches, Validation dataset %d batches, batch size %d" %
              (len(train_loader), len(valid_loader), args.batch_size))

        # Print twice in one epoch
        print('Starting training...')
        train_logging_step = (len(train_loader) + 3) // 3
        loc_logs_train, current_train_loss = train_step(
            train_loader,
            cl_model,
            cl_transform,
            cl_optimizer,
            update_label=not warm_up_stage,
            epoch_num=epoch,
            logging_step=train_logging_step,
        )
        print('Starting validation...')
        valid_logging_step = (len(valid_loader) + 3) // 3
        loc_logs_valid, current_valid_loss, current_valid_f1 = valid_step(
            valid_loader,
            cl_model,
            cl_transform,
            logging_step=valid_logging_step,
        )

        total_change_num = 0
        if not warm_up_stage:
            train_dataset.update_correct_label(
                cl_transform.get_correct_label()
            )

            print('Starting check clean training labels...')
            total_change_num = change_step(
                ori_y,
                train_dataset.label,
            )

        print(f'Ran {epoch - start_epoch + 1} epochs in {time.time() - main_start_time:.2f} seconds')

        # When all samples are fully trained, we compute the convergence and early stop condition
        stop_flag = \
            (epoch >= (args.warm_epoch_num + (args.num_level - 1) * args.level_gap_epoch_num + args.cl_epoch_num))
        if stop_flag:
            # process training loss
            loss_change = np.fabs(current_train_loss - last_loss)
            last_loss = current_train_loss
        else:
            loss_change = args.loss_change + 1

        # process validation indexes
        early_flag = True
        if current_valid_loss < best_valid_loss:
            best_valid_loss = deepcopy(current_valid_loss)
            best_loss_model_state = deepcopy(cl_model.state_dict())
            print('Loss Bingo!!!')
            wait_epoch = 0
            early_flag = False
        if current_valid_f1 > best_valid_f1:
            best_valid_f1 = deepcopy(current_valid_f1)
            best_f1_model_state = deepcopy(cl_model.state_dict())
            print('F1 Bingo!!!')
            wait_epoch = 0
            early_flag = False
        if max_total_change_num < total_change_num:
            max_total_change_num = total_change_num
            wait_epoch = 0
            early_flag = False
        if args.early_stop and stop_flag and early_flag:
            wait_epoch += 1
        print(f'Waiting Epoch: {wait_epoch}')

        # We record the best results across all patience epochs
        # save_correct_label can be saved for training other models
        if args.best_val_index == 'loss':
            last_valid_loss.pop(0)
            if current_valid_loss < min(last_valid_loss):
                last_model_state = deepcopy(cl_model.state_dict())
                save_correct_label = train_dataset.label
                print(f'Last loss Bingo!!!\n{last_valid_loss}')
            last_valid_loss.append(current_valid_loss)
        else:
            last_valid_f1.pop(0)
            if current_valid_f1 > max(last_valid_f1):
                last_model_state = deepcopy(cl_model.state_dict())
                save_correct_label = train_dataset.label
                print(f'Last F1 Bingo!!!\n{last_valid_f1}')
            last_valid_f1.append(current_valid_f1)

        main_logs = utils.main_logs_update(main_logs, loc_logs_train, loc_logs_valid, epoch)

        if path_checkpoint is not None and (
                epoch % args.save_step == 0 or epoch == args.all_epoch_num - 1 or
                loss_change <= args.loss_change or wait_epoch >= args.patience):
            optimizer_state = cl_optimizer.state_dict()

            utils.save_checkpoint(
                last_model_state,
                optimizer_state,
                best_loss_model_state,
                best_f1_model_state,
                best_valid_loss,
                best_valid_f1,
                f"{path_checkpoint}_0.pt",
            )
            utils.save_logs(main_logs, f"{path_checkpoint}_logs.json")

        if loss_change <= args.loss_change or wait_epoch >= args.patience:
            break

    train_dataset.reload_pool.close()
    valid_dataset.reload_pool.close()

    print('-' * 10, 'CL Training finished', '-' * 10)
