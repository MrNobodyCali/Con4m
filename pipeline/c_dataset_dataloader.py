import torch
import torch.nn.functional as F
from torch.multiprocessing import Pool
from torch.utils.data import Dataset, DataLoader

from ca_database_api import DataHandler


class CLDataSet(Dataset):
    def __init__(
            self,
            args,
    ):
        print("Loading the dataset...")
        self.data_handler = DataHandler(
            database_save_dir=args.database_save_dir,
            data_name=args.data_name,
            patient_list=args.patient_list,
            noise_ratio=args.noise_ratio,
            window_time=args.window_time,
            slide_time=args.slide_time,
            num_level=5,
        )
        data_pack = self.data_handler.get_data()

        # data.size(): num_level x seg_big_num x seg_small_num x n_features x length
        self.data = torch.tensor(data_pack.data, dtype=torch.float)
        # label.size(): num_level x seg_big_num x seg_small_num
        self.label = torch.tensor(data_pack.label, dtype=torch.long)
        self.class_name = self.label.unique()
        self.n_class = len(self.class_name)
        # loc.size(): num_level x seg_big_num x seg_small_num x 3
        self.loc = torch.tensor(data_pack.loc, dtype=torch.long)

        # label.size(): n_level x seg_big_num x seg_small_num x n_class
        self.label = F.one_hot(self.label, num_classes=self.n_class)
        self.sample_ratio = self.label.view(-1, self.n_class).float().mean(dim=0)
        print('Class weight is:', self.sample_ratio)

        self.patient_list = args.patient_list
        self.seg_small_num = self.label.size(-2)
        self.n_features = self.data.size(-2)
        self.warm_epoch = args.warm_epoch_num
        self.nProcessLoader = args.n_process_loader
        self.reload_pool = Pool(args.n_process_loader)

        # For curriculum learning
        self.num_level = args.num_level
        print('data.size():', self.data.size())

        self.level_gap_epoch = args.level_gap_epoch_num
        self.epoch_num = -1

        self.cl_epoch = args.cl_epoch_num
        self.eta_add = 1. / self.cl_epoch
        self.eta_add_condition = torch.arange(0, self.num_level * self.level_gap_epoch,
                                              self.level_gap_epoch) + self.warm_epoch
        self.eta = torch.zeros(self.num_level)

    def __len__(self):
        if self.epoch_num < self.warm_epoch:
            return self.num_level * self.data.size(1)
        level_num = (self.epoch_num >= self.eta_add_condition).sum()
        return level_num * self.data.size(1)

    def __getitem__(self, index):
        # The level data is organized in training order
        level_num = index // self.data.size(1)
        big_seg_num = index % self.data.size(1)

        return self.data[level_num, big_seg_num], self.label[level_num, big_seg_num], self.eta[[level_num]], \
            torch.tensor([level_num, big_seg_num])

    def get_data_loader(self, batch_size, shuffle=False, num_workers=0):
        self.epoch_num += 1
        self.eta += self.eta_add * (self.epoch_num > self.eta_add_condition)
        self.eta = torch.min(self.eta, torch.ones_like(self.eta))
        return DataLoader(
            self,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
        )

    def get_initial_label(self):
        if self.epoch_num < self.warm_epoch:
            return self.label
        else:
            return None

    def update_correct_label(self, label):
        self.label = label.clone()
