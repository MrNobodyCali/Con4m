import random
import torch
import numpy as np
from copy import deepcopy
import sys
import time
import os
import psutil
import json


def update_logs(logs, log_step, prev_logs=None):
    out = {}
    for key in logs:
        out[key] = deepcopy(logs[key])

        if prev_logs is not None:
            out[key] -= prev_logs[key]
        out[key] /= log_step
    return out


def show_logs(text, logs):
    print("")
    print('-'*50)
    print(text)

    for key in logs:
        if key in ["train_iter", "valid_iter"]:
            continue

        n_predicts = logs[key].shape[0]

        str_steps = ['Step'] + [str(s) for s in range(1, n_predicts + 1)]
        format_command = ' '.join(['{:>16}' for x in range(n_predicts + 1)])
        print(format_command.format(*str_steps))

        str_log = [key] + ["{:10.6f}".format(s) for s in logs[key]]
        print(format_command.format(*str_log))

    print('-'*50)


def main_logs_update(logs, loc_logs_train, loc_logs_val, epoch):
    for key, value in dict(loc_logs_train, **loc_logs_val).items():
        if key not in logs:
            logs[key] = []
        if isinstance(value, np.ndarray):
            value = value.tolist()
        logs[key].append(value)
    logs["epoch"].append(epoch)
    return logs


def batch_logs_update(task_type, logs, last_logs, loss, acc,
                      step=None, logging_step=None, start_time=None, n_examples=None, show_string=''):
    if "locLoss_" + task_type not in logs:
        if loss.ndim == 0:
            logs["locLoss_" + task_type] = np.zeros(1)
            logs["locAcc_" + task_type] = np.zeros(1)
        else:
            # Multiple steps
            logs["locLoss_" + task_type] = np.zeros(loss.size(0))
            logs["locAcc_" + task_type] = np.zeros(acc.size(0))

    logs["locLoss_" + task_type] += loss.detach().cpu().numpy()
    logs["locAcc_" + task_type] += acc.cpu().numpy()

    if step is not None:
        if (step + 1) % logging_step == 0:
            new_time = time.perf_counter()
            elapsed = new_time - start_time
            print(f"Update {step + 1}")
            print(f"elapsed: {elapsed:.1f} s")
            print(
                f"{1000.0 * elapsed / logging_step:.1f} ms per batch, {1000.0 * elapsed / n_examples:.1f} ms / example")
            loc_logs = update_logs(logs, logging_step, last_logs)
            last_logs = deepcopy(logs)
            show_logs(show_string + " " + task_type + " loss", loc_logs)
        return last_logs
    return logs


def save_logs(data, path_logs):
    with open(path_logs, 'w') as file:
        json.dump(data, file, indent=2)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cpu_stats():
    print(sys.version)
    print(psutil.cpu_percent())
    print(psutil.virtual_memory())


def load_checkpoint(load_path):
    if not os.path.isdir(load_path):
        if load_path.split('.')[-1] == 'pt':
            checkpoint_file = load_path
            load_path = os.path.split(load_path)[0]
        else:
            print("Invalid checkpoints path at " + load_path)
            return None
    else:
        checkpoints = [x for x in os.listdir(load_path)
                       if os.path.splitext(x)[1] == '.pt'
                       and os.path.splitext(x[11:])[0].isdigit()]
        if len(checkpoints) == 0:
            print("No checkpoints found at " + load_path)
            return None
        checkpoints.sort(key=lambda x: int(os.path.splitext(x[11:])[0]))
        checkpoint_file = os.path.join(load_path, checkpoints[-1])
    with open(os.path.join(load_path, 'checkpoint_logs.json'), 'rb') as file:
        logs = json.load(file)

    return os.path.abspath(checkpoint_file), logs


def save_checkpoint(model_state, optimizer_state, best_loss_model_state, best_f1_model_state,
                    best_val_loss, best_val_f1, path_checkpoint):
    state_dict = {
        "CLModel": model_state,
        "Optimizer": optimizer_state,
        "BestLossModel": best_loss_model_state,
        "BestF1Model": best_f1_model_state,
        "BestValLoss": best_val_loss,
        "BestValF1": best_val_f1,
    }

    torch.save(state_dict, path_checkpoint)
