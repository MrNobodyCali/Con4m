import os
import sys
import math
import argparse
import numpy as np

from a_data_file_load import FNIRSDataLoader


def extract_database(
        root_path,          # The root path of the raw files
        save_dir,           # The save file holder of the database
        data_name,          # The name of database
        seg_len,            # The length of time intervals
        seg_num,            # The number of time intervals for each level, each class and each subject
        num_class,          # The number of classes
        noise_ratio,        # The ratio of adding noise
        num_level,          # The number of levels
):
    print('Reading data from:', root_path)
    raw_data_loader = FNIRSDataLoader(root_path=root_path)
    x = []
    y = []
    num_subject = raw_data_loader.num_subject
    print('num_subject:', num_subject)
    for s_id in range(1, num_subject + 1):
        data, label = raw_data_loader.read(s_id)
        if data_name == 'fNIRS_2':
            data, label = __process_to_two_classes__(data, label)
        x.append(data)
        y.append(label)

    print('-' * 10, 'Random splitting the subjects', '-' * 10)
    group_list = __random_split_subject__(num_subject)

    for r in noise_ratio:
        print('-' * 10, f'Adding {r} noise for original labels', '-' * 10)
        if r == 0:
            noise_y = y
        else:
            noise_y = __add_noise__(y, r)

        for g_id in range(len(group_list)):
            g_x = [x[i] for i in group_list[g_id]]
            g_noisy_y = [noise_y[i] for i in group_list[g_id]]

            print('-' * 10, f'Processing database for group {g_id}', '-' * 10)
            print('-' * 10, 'Decompose the labels for different classes and levels', '-' * 10)
            total_file_list, total_start_num_list = \
                __decompose_data__(
                    label=g_noisy_y,
                    seg_len=seg_len,
                    num_class=num_class,
                    num_level=num_level,
                )
            print('-' * 10, 'Decompose data done', '-' * 10)

            print('-' * 10, 'Sampling the segment database for different classes and levels', '-' * 10)
            total_seg_list = __sampling_segment__(
                start_num_list=total_start_num_list,
                seg_len=seg_len,
                seg_num=seg_num * len(group_list[g_id]),
                num_class=num_class,
                num_level=num_level,
            )
            print('-' * 10, 'Sampling segments done', '-' * 10)

            print('-' * 10, 'Saving database', '-' * 10)
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
        print('-' * 10, f'Saving done for noise ratio {r}', '-' * 10)


def __process_to_two_classes__(x, y):
    # Only include class-0 and class-2
    valid_index = np.where((y == 0) | (y == 2))[0]
    x = x[valid_index]
    y = y[valid_index]
    assert len(np.unique(y)) == 2
    y = (y == 2).astype(np.int64)
    return x, y


def __add_noise__(
        y,
        noise_ratio,
):
    noise_y = []
    for s_id in range(len(y)):
        tmp_y = y[s_id].copy()

        # Find continuous big states
        boundary_index = np.where(np.diff(tmp_y))[0]
        # The file only includes one single class (No boundary)
        if len(boundary_index) == 0:
            print('The file only includes one class and no noises.')
            print('-' * 50)
            noise_y.append(tmp_y)
            continue
        length = np.diff(np.append(boundary_index, len(tmp_y) - 1))
        length = np.append(boundary_index[0] + 1, length)

        # Decide to change which direction
        direction = np.random.rand(len(boundary_index)) >= 0.5
        print('direction:\n', list(direction))
        # Decide to add how many noise
        relative_index_list = []
        for seg_id in range(len(boundary_index)):
            b_index = boundary_index[seg_id]
            # Forward/Right
            if direction[seg_id]:
                # Maybe the noisy length is zero, then we overlook this noise
                if int(length[seg_id + 1] * noise_ratio) == 0:
                    relative_index = 0
                else:
                    relative_index = np.random.choice(int(length[seg_id + 1] * noise_ratio))
                tmp_y[b_index + 1:b_index + 1 + relative_index] = tmp_y[b_index]
            # Backward/Left
            else:
                # Maybe the noisy length is zero, then we overlook this noise
                if int(length[seg_id] * noise_ratio) == 0:
                    relative_index = 0
                else:
                    relative_index = np.random.choice(int(length[seg_id] * noise_ratio))
                tmp_y[b_index + 1 - relative_index:b_index + 1] = tmp_y[b_index + 1]
            relative_index_list.append(relative_index)
        print('relative_index_list:\n', relative_index_list)
        print('-' * 50)
        noise_y.append(tmp_y)
    return noise_y


# Extract the candidate big states for sampling
def __decompose_data__(
        label,
        seg_len,
        num_class,
        num_level,
):
    # Record the cut list of every file for each level in each class
    total_file_list = [[[] for _ in range(num_level)] for _ in range(num_class)]
    # Record the candidate start points number list of every file for each level in each class
    total_start_num_list = [[[] for _ in range(num_level)] for _ in range(num_class)]

    # Extract all the big segments
    for s_id in range(len(label)):
        y = label[s_id]

        for c in range(num_class):
            total_file = [[] for _ in range(num_level)]
            total_start_num = [[] for _ in range(num_level)]

            # Find the big cuts
            label_index = np.where(y == c)[0]
            # Have at least one cut
            if len(label_index) != 0:
                slot_pair_index = compute_slot(label_index)
                for s, e in slot_pair_index:
                    old_start_ = label_index[s]
                    old_end_ = label_index[e]

                    start_, end_ = boundary_detection(old_start_, old_end_, seg_len, y)
                    for (start, end) in [(start_, old_end_ - (seg_len - seg_len // 2) + 1),
                                         (old_start_ + seg_len // 2, end_)]:
                        # Verify the cut has positive length
                        if end - start >= 4 * num_level:
                            assert 1 <= len(np.unique(np.diff(y[start - seg_len // 2:end + (
                                        seg_len - seg_len // 2)]))) <= 2
                            # Divide into some levels from middle to edges
                            middle_point = start + (end - start) / 2.
                            radius = (end - start) / (2. * num_level)
                            # record the boundary of the last level
                            left_b, right_b = int(middle_point), int(middle_point) + 1
                            for level in range(num_level):
                                # add left half points
                                total_file[level].append([math.ceil(middle_point - radius * (level + 1)) + 1,
                                                          left_b])
                                total_start_num[level].append(total_file[level][-1][-1] - total_file[level][-1][-2] + 1)
                                assert total_start_num[level][-1] > 0
                                left_b = total_file[level][-1][-2] - 1

                                # add right half points
                                total_file[level].append([right_b,
                                                          math.floor(middle_point + radius * (level + 1))])
                                total_start_num[level].append(total_file[level][-1][-1] - total_file[level][-1][-2] + 1)
                                assert total_start_num[level][-1] > 0
                                right_b = total_file[level][-1][-1] + 1

            for level in range(num_level):
                total_file_list[c][level].append(total_file[level])
                total_start_num_list[c][level].append(total_start_num[level])

    for c in range(num_class):
        print('-' * 10, 'Results for Class', c, '-' * 10)
        for level in range(num_level):
            print('-' * 10, 'Results for Level', level, '-' * 10)
            print('total_file_list:\n', total_file_list[c][level])
            print('total_start_num_list:\n', total_start_num_list[c][level])

    return total_file_list, total_start_num_list


def __sampling_segment__(
        start_num_list,
        seg_len,
        seg_num,
        num_class,
        num_level,
):
    # The return list
    total_seg_list = []

    # Sample for each class separately
    for c in range(num_class):
        total_seg = [[] for _ in range(num_level)]
        # Sample for each level separately
        for level in range(num_level):
            # Randomly select the middle points of segments
            total_num = 0
            for file_index in range(len(start_num_list[c][level])):
                for seg_index in range(len(start_num_list[c][level][file_index])):
                    total_num += start_num_list[c][level][file_index][seg_index]
            random_list = np.random.choice(total_num, seg_num)

            for index in random_list:
                index_copy = index
                bingo = False
                for file_index in range(len(start_num_list[c][level])):
                    for seg_index in range(len(start_num_list[c][level][file_index])):
                        if index_copy >= start_num_list[c][level][file_index][seg_index]:
                            index_copy -= start_num_list[c][level][file_index][seg_index]
                        else:
                            total_seg[level].append([file_index, seg_index,
                                                     index_copy - seg_len // 2, index_copy + (seg_len - seg_len // 2)])
                            bingo = True
                            break
                    if bingo:
                        break

        total_seg_list.append(total_seg)

    for c in range(num_class):
        print('-' * 10, 'Results for Class', c, '-' * 10)
        for level in range(num_level):
            print('-' * 10, 'Results for Level', level, '-' * 10)
            print('total_seg_list:\n', total_seg_list[c][level])

    return total_seg_list


def __save_data__(
        data,
        label,
        s_id,
        noise_ratio,
        file_list,
        seg_list,
        num_class,
        num_level,
        save_dir,
):
    save_dir = os.path.join(save_dir, str(int(noise_ratio * 100)))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data_save = [[] for _ in range(num_level)]
    label_save = [[] for _ in range(num_level)]
    # element in loc_save: [file_num, start, end]
    loc_save = [[] for _ in range(num_level)]

    class_seg_count = [0 for _ in range(num_class)]
    for c in range(num_class):
        for level in range(num_level):
            for meta_tuple in seg_list[c][level]:
                file_index, seg_index, r_start, r_end = meta_tuple
                start, _ = file_list[c][level][file_index][seg_index]
                s, e = start + r_start, start + r_end

                # Self-check the illegal labels
                unique_label = set(np.unique(label[file_index][s:e]))
                if not unique_label.issubset(set(range(num_class))):
                    raise ValueError('There exist illegal labels', unique_label, 'from', file_index, s, e)
                assert 1 <= len(unique_label) <= 2
                assert 1 <= len(np.unique(np.diff(label[file_index][s:e]))) <= 2
                main_class = np.argmax(np.bincount(label[file_index][s:e]))
                class_seg_count[main_class] += 1

                label_save[level].append(label[file_index][s:e])
                loc_save[level].append([file_index, s, e])
                data_save[level].append(data[file_index][s:e])

    print('The segment number of each class:', class_seg_count)
    for level in range(num_level):
        np.savez_compressed(
            os.path.join(save_dir, f's{s_id}_level{level}_sample.npz'),
            data=np.array(data_save[level]),
            label=np.array(label_save[level]),
            loc=np.array(loc_save[level])
        )


def __random_split_subject__(num_subject, num_split=4):
    group_num = num_subject // num_split

    random_split = np.random.choice(num_subject, num_subject, replace=False)
    g_list = []
    for g_id in range(1, num_split + 1):
        group = random_split[(g_id - 1) * group_num:g_id * group_num]
        g_list.append(list(group))
        print(f"'g{g_id}': {g_list[-1]},")
    return g_list


def compute_slot(index, boundary=1):
    end_index = np.where(np.diff(index) > boundary)[0]
    start_index = end_index + 1
    end_index = np.append(end_index, len(index) - 1)
    start_index = np.append(0, start_index)

    return list(zip(start_index, end_index))


def boundary_detection(start, end, seg_len, label):
    # Avoid to be out of the start file bound
    if start - seg_len // 2 < 0:
        start = seg_len // 2
    # Avoid to be out of the end file bound
    if end + (seg_len - seg_len // 2 - 1) >= len(label):
        end = len(label) - (seg_len - seg_len // 2 - 1) - 1

    # Remove the same label crossing two labels in the left bound
    left_label = label[start - 1]
    t = np.nonzero(label[start - seg_len // 2:start] != left_label)[0]
    if len(t) != 0:
        start += t.max() + 1
    # Remove the same label crossing two labels in the right bound
    right_label = label[end + 1]
    t = np.nonzero(label[end + 1:end + 1 + (seg_len - seg_len // 2 - 1)] != right_label)[0]
    if len(t) != 0:
        end -= (seg_len - seg_len // 2 - 1) - t.min()

    return start, end


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Database')
    parser.add_argument('--load_dir', type=str, default='/data/eeggroup/public_dataset/',
                        help='Should give an absolute path including all the original data files.')
    parser.add_argument('--save_dir', type=str, default='/data/eeggroup/CL_database/',
                        help='Should give an absolute path to save the database.')
    parser.add_argument('--data_name', type=str, default='fNIRS_2',
                        help='Should give the name of the public database.')
    parser.add_argument('--noise_ratio', nargs='*', type=float, default=None,
                        help='The ratio list of adding noise.')
    parser.add_argument('--seg_len', type=int, default=40 * 5,
                        help='The number of time points of sampled time intervals.')
    parser.add_argument('--num_level', type=int, default=5,
                        help='The number of levels.')
    argv = sys.argv[1:]
    args = parser.parse_args(argv)

    load_dir_dict = {
        'fNIRS': 'Tufts_fNIRS_data',
        'fNIRS_2': 'Tufts_fNIRS_data',
    }

    args.load_dir = os.path.join(args.load_dir, load_dir_dict[args.data_name])

    seg_num_dict = {
        'fNIRS': 3,
        'fNIRS_2': 6,
    }
    num_class_dict = {
        'fNIRS': 4,
        'fNIRS_2': 2,
    }

    args.seg_num = seg_num_dict[args.data_name]
    args.num_class = num_class_dict[args.data_name]
    if args.noise_ratio is None:
        args.noise_ratio = [.0, .1, .2, .3, .4]
    args.save_dir = os.path.join(args.save_dir, args.data_name)

    extract_database(
        root_path=args.load_dir,
        save_dir=args.save_dir,
        data_name=args.data_name,
        seg_len=args.seg_len,
        seg_num=args.seg_num,
        num_class=args.num_class,
        noise_ratio=args.noise_ratio,
        num_level=args.num_level,
    )
    print('-' * 10, 'ALL Done', '-' * 10)
