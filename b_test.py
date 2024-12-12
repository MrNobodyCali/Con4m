import os
import sys
import json
import time
import torch
import argparse

sys.path.append('utils')
sys.path.append('pipeline')
sys.path.append('model')

import utils.misc as utils
from utils.default_config import set_default_config, get_choice_default_config
from utils.excel_manager import ExcelManager
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


def test_step(
        dataloader,
        model,
        transform,
        logging_step,
):
    model.eval()
    device = next(model.parameters()).device
    start_time = time.perf_counter()
    n_examples = 0

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

        true_label = torch.cat((true_label, torch.argmax(batch_label, dim=-1).view(-1).cpu()))
        pred_label = torch.cat((pred_label, c_pred.view(-1).cpu()))

        if (step + 1) % logging_step == 0:
            new_time = time.perf_counter()
            elapsed = new_time - start_time
            print(f"Update {step + 1}")
            print(f"elapsed: {elapsed:.1f} s")
            print(f"{1000.0 * elapsed / (logging_step * (step + 1)):.1f} ms per batch, "
                  f"{1000.0 * elapsed / n_examples:.1f} ms / example")
            print('-' * 50)

    print(f'Testing time: {time.perf_counter() - start_time:.2f} seconds')
    index = dataloader.dataset.data_handler.model_evaluation(
        true_label.numpy(),
        pred_label.numpy(),
        dataloader.dataset.n_class,
    )
    print('-' * 10, 'The average testing results', '-' * 10)
    print(index)
    return index


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLModel')
    parser = set_default_config(parser)

    group_test = parser.add_argument_group('Test')
    group_test.add_argument('--database_save_dir', type=str, default='/data/eeggroup/CL_database/',
                            help='Should give a path to load the database.')
    group_test.add_argument('--data_name', type=str, default='fNIRS_2',
                            help='Should give the name of the database [fNIRS_2, Sleep, HHAR].')
    group_test.add_argument('--noise_ratio', type=float, default=.0,
                            help='The noisy ratio of the loading model.')
    group_test.add_argument('--exp_id', type=int, default=1,
                            help='The experimental id. fNIRS_2: 1-12; Sleep/HHAR: 1-6.')
    group_test.add_argument('--gpu_id', type=int, default=0,
                            help='The gpu id.')
    group_test.add_argument('--batch_size', type=int, default=64,
                            help='Number of testing batches.')
    group_test.add_argument('--load_path', type=str, default='/data/eeggroup/CL_result/',
                            help='The path to load checkpoint.')
    group_test.add_argument('--gpu', action='store_false',
                            help='Whether to load the data and model to GPU.')
    group_test.add_argument('--summary', type=bool, default=False,
                            help='Whether to summary the results of all experiments.')

    argv = sys.argv[1:]
    args = parser.parse_args(argv)
    args, config = get_choice_default_config(args)

    args.load_path = os.path.join(args.load_path, f'{args.data_name}/Con4m/')
    args.load_path = os.path.join(args.load_path, f'{int(args.noise_ratio * 100)}/')
    args.load_path = os.path.join(args.load_path, f'exp{args.exp_id}')

    # Save to Excel file
    excel = ExcelManager(args.load_path, 'test_result')
    # Aggregate all experimental results into one Excel file
    if args.summary:
        excel.summary_results()
        sys.exit(0)

    # Extra parameters
    args.patient_list = args.test_patient_list
    # Testing dataset should be clean
    args.noise_ratio = 0

    test_dataset = CLDataSet(args)
    # Make sure to use all level data
    test_dataset.epoch_num = 200

    config.n_class = test_dataset.n_class
    config.seg_small_num = test_dataset.seg_small_num
    config.raw_input_len = test_dataset.data_handler.window_len
    config.n_features = test_dataset.n_features

    if args.load_path is not None:
        load_path, _ = utils.load_checkpoint(args.load_path)
        print('-' * 50)
        print('Load checkpoint:', load_path)

        state_dict = torch.load(load_path, 'cpu')
        best_loss_model = state_dict["BestLossModel"]
        best_f1_model = state_dict["BestF1Model"]
        last_model = state_dict["CLModel"]
    else:
        raise ValueError('--load_path cannot be None.')

    model_name = ['loss', 'f1', 'last']
    for i, model_dict in enumerate([best_loss_model, best_f1_model, last_model]):
        cl_model = CLModel(config)
        cl_transform = CLTransform(
            ori_y=torch.zeros(1),
            batch_size=args.batch_size,
            seg_num=config.seg_small_num,
        )

        cl_model.load_state_dict(model_dict, strict=False)

        if args.gpu:
            gpu_device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
            cl_model.to(gpu_device)
            cl_transform.l_tanh.to(gpu_device)

        print(f'CONFIG:\n{json.dumps(vars(args), indent=4, sort_keys=True)}')
        print('-' * 50)
        utils.cpu_stats()

        test_loader = test_dataset.get_data_loader(args.batch_size, shuffle=False, num_workers=0)
        print("Testing dataset %d batches, batch size %d" % (len(test_loader), args.batch_size))

        test_logging_step = (len(test_loader) + 3) // 3
        infer_index = test_step(
            test_loader,
            cl_model,
            cl_transform,
            logging_step=test_logging_step,
        )

        excel.res2excel(str(infer_index), tar_pat_name=f'exp{args.exp_id}_{model_name[i]}')

        test_dataset.reload_pool.close()

    # Save single experimental result
    excel.excel_save(args.exp_id)
    print('-' * 10, 'CL Testing finished', '-' * 10)
