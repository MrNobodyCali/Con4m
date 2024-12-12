import os
import sys
import argparse

from a_data_file_load import HHARDataLoader
from b_fNIRS_database_generate import __add_noise__, __decompose_data__, __sampling_segment__, __save_data__


def extract_database(
        root_path,          # The root path of the raw files
        save_dir,           # The save file holder of the database
        noise_ratio,        # The ratio of adding noise
        seg_len,            # The length of time intervals
        seg_num,            # The number of time intervals for all levels and all classes
        num_class,          # The number of classes
        num_level,          # The number of levels
):
    print('Reading data from:', root_path)
    raw_data_loader = HHARDataLoader(path=root_path)
    x = []
    y = []
    num_group = raw_data_loader.num_group
    print('num_group:', num_group)
    for g_id in range(num_group):
        data, label = raw_data_loader.read(ids_idx=g_id)
        x.append(data)
        y.append(label)
    seg_len = int(seg_len * raw_data_loader.sample_rate)

    group_list = [[i] for i in range(num_group)]

    seg_num = ((seg_num // num_class) // num_level) // num_group
    group_name = list(range(num_group))
    print('seg_len:', seg_len)
    print('seg_num:', seg_num)

    for r in noise_ratio:
        print('-' * 10, f'Adding {r} noise for original labels', '-' * 10)
        if r == 0:
            noise_y = y
        else:
            noise_y = __add_noise__(y, r)

        for g_id in group_name:
            g_x = [x[i] for i in group_list[g_id]]
            g_noisy_y = [noise_y[i] for i in group_list[g_id]]

            print('-' * 10, f'Decompose the {g_id} labels for different classes and levels', '-' * 10)
            total_file_list, total_start_num_list = \
                __decompose_data__(
                    label=g_noisy_y,
                    seg_len=seg_len,
                    num_class=num_class,
                    num_level=num_level,
                )
            print('-' * 10, 'Decompose data done', '-' * 10)

            print('-' * 10, f'Sampling the segments for {g_id} database', '-' * 10)
            total_seg_list = __sampling_segment__(
                start_num_list=total_start_num_list,
                seg_len=seg_len,
                seg_num=seg_num,
                num_class=num_class,
                num_level=num_level,
            )
            print('-' * 10, 'Sampling segments done', '-' * 10)

            print('-' * 10, f'Saving {g_id} database', '-' * 10)
            __save_data__(
                data=g_x,
                label=g_noisy_y,
                s_id=g_id,
                noise_ratio=r,
                file_list=total_file_list,
                seg_list=total_seg_list,
                num_class=num_class,
                num_level=num_level,
                save_dir=save_dir,
            )
            print('-' * 50)
        print('-' * 10, f'Saving done for noise ratio {r}', '-' * 10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Database')
    parser.add_argument('--load_dir', type=str, default='/data/eeggroup/public_dataset/',
                        help='Should give an absolute path including all the original data files.')
    parser.add_argument('--save_dir', type=str, default='/data/eeggroup/CL_database/',
                        help='Should give an absolute path to save the database.')
    parser.add_argument('--data_name', type=str, default='HHAR',
                        help='Should give the name of the public database.')
    parser.add_argument('--noise_ratio', nargs='*', type=float, default=None,
                        help='The ratio list of adding noise.')
    parser.add_argument('--seg_len', type=int, default=60,
                        help='The seconds of sampled time intervals.')
    parser.add_argument('--seg_num', type=int, default=5400,
                        help='The total number of intervals to sample for all levels and all classes.')
    parser.add_argument('--num_class', type=int, default=6,
                        help='The number of classes.')
    parser.add_argument('--num_level', type=int, default=5,
                        help='The number of levels.')
    argv = sys.argv[1:]
    args = parser.parse_args(argv)

    load_dir_dict = {
        'HHAR': 'hhar',
    }

    args.load_dir = os.path.join(args.load_dir, load_dir_dict[args.data_name])
    if args.noise_ratio is None:
        args.noise_ratio = [.0, .1, .2, .3, .4]
    args.save_dir = os.path.join(args.save_dir, args.data_name)

    extract_database(
        root_path=args.load_dir,
        save_dir=args.save_dir,
        noise_ratio=args.noise_ratio,
        seg_len=args.seg_len,
        seg_num=args.seg_num,
        num_class=args.num_class,
        num_level=args.num_level,
    )
    print('-' * 10, 'ALL Done', '-' * 10)
