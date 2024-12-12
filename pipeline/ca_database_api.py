import os
import numpy as np

from cb_evaluation_api import class_evaluation


class DataPack:
    def __init__(self):
        # basic content
        self.data = None
        self.label = None

        # used to locate the data
        self.loc = None
        self.patient_list = None


class DataHandler:
    def __init__(
            self,
            database_save_dir='/data/eeggroup/CL_database/',     # The path of database file holder
            data_name='fNIRS_2',                                 # The name of database
            patient_list=None,      # The list of patient group
            noise_ratio=None,       # The noise ratio for the original labels
            window_time=1,          # The unit is second except for fNIRS_2
            slide_time=0.5,         # The unit is second except for fNIRS_2
            num_level=5,            # The number of levels
    ):
        assert data_name in ['fNIRS_2', 'Sleep', 'HHAR']
        self.database_save_dir = os.path.join(database_save_dir, data_name)
        self.data_name = data_name
        self.patient_list = patient_list
        self.noise_ratio = noise_ratio

        if data_name in ['Sleep', 'HHAR']:
            if data_name == 'Sleep':
                self.sample_rate = 100
            else:
                self.sample_rate = 50
            self.window_len = int(window_time * self.sample_rate)
            self.slide_len = int(slide_time * self.sample_rate)
        else:
            self.sample_rate = None
            self.window_len = int(window_time)
            self.slide_len = int(slide_time)

        self.num_level = num_level

    def obtain_database_dir(self, pa, level):
        pre_fix = f'{int(self.noise_ratio * 100)}/s{pa}_level{level}_sample'
        return os.path.join(self.database_save_dir, pre_fix + '.npz')

    def __get_database__(self):
        data_pack = DataPack()
        data_pack.data = [[] for _ in range(self.num_level)]
        data_pack.label = [[] for _ in range(self.num_level)]
        data_pack.loc = [[] for _ in range(self.num_level)]
        data_pack.patient_list = self.patient_list

        for pa in self.patient_list:
            for level in range(self.num_level):
                load_path = self.obtain_database_dir(pa, level)
                print(f'Loading the labels from: {load_path}')
                all_data = np.load(load_path)

                data_pack.data[level].append(all_data['data'])
                data_pack.label[level].append(all_data['label'])
                data_pack.loc[level].append(all_data['loc'])

        # num_level x (pa * sample_num) x length x n_features
        data_pack.data = np.array(data_pack.data).reshape([self.num_level, -1, *data_pack.data[0][0].shape[-2:]])
        data_pack.label = np.array(data_pack.label).reshape([self.num_level, -1, data_pack.label[0][0].shape[-1]])
        data_pack.loc = np.array(data_pack.loc).reshape([self.num_level, -1, data_pack.loc[0][0].shape[-1]])

        print('Total BIG Segment Number for', self.patient_list, ':', data_pack.data.shape[0] * data_pack.data.shape[1])
        return self.get_segment_data(data_pack)

    def get_data(self):
        return self.__get_database__()

    def get_segment_data(self, data_pack):
        # num_level x sample_num x length x n_features
        data, label, loc = data_pack.data, data_pack.label, data_pack.loc

        new_data_pack = DataPack()
        new_data_pack.data = []
        new_data_pack.label = []
        new_data_pack.loc = []
        new_data_pack.patient_list = data_pack.patient_list

        start = np.arange(0, data.shape[-2] - self.window_len + 1, self.slide_len)
        end = start + self.window_len
        start_end_pair = list(zip(start, end))

        for s, e in start_end_pair:
            new_data_pack.data.append(data[:, :, s:e])
            new_data_pack.label.append(label[:, :, s:e])
            tmp_loc = loc.copy()
            tmp_loc[:, :, -1] = tmp_loc[:, :, -2] + e
            tmp_loc[:, :, -2] += s
            new_data_pack.loc.append(tmp_loc)

        # segment_num x num_level x sample_num x window_size x n_features ->
        # num_level x sample_num x segment_num x window_size x n_features
        new_data_pack.data = np.stack(new_data_pack.data, axis=-3)
        new_data_pack.data = new_data_pack.data.transpose(0, 1, 2, 4, 3)
        # num_level x sample_num x segment_num
        new_data_pack.label = np.stack(new_data_pack.label, axis=-2)
        max_label = new_data_pack.label.max(axis=-1)
        min_label = new_data_pack.label.min(axis=-1)
        mean_label = new_data_pack.label.mean(axis=-1)
        flag_label = (mean_label >= (max_label + min_label) / 2)
        new_data_pack.label = max_label * flag_label + min_label * ~flag_label
        # num_level x sample_num x segment_num x 3
        new_data_pack.loc = np.stack(new_data_pack.loc, axis=-2)

        print('SMALL Segment Number for each big segment:', new_data_pack.data.shape[-3])
        return new_data_pack

    @staticmethod
    def model_evaluation(true_label, pred_label, n_class):
        return class_evaluation(true_label, pred_label, n_class)
