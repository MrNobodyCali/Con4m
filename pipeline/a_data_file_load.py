import os
import mne
import csv
import glob
import h5py
import numpy as np
from mne.io import read_raw_edf
from sklearn.preprocessing import scale


class FNIRSDataLoader:
    def __init__(self, root_path='/data/eeggroup/public_dataset/Tufts_fNIRS_data/'):
        self.root_path = root_path
        self.tag = ['AB_I_O', 'AB_PHI_O', 'AB_I_DO', 'AB_PHI_DO', 'CD_I_O', 'CD_PHI_O', 'CD_I_DO', 'CD_PHI_DO', 'label']

        self.file_name_list = []
        for file in os.listdir(self.root_path):
            if file.split('.')[-1] == 'csv':
                self.file_name_list.append(file)
        self.num_subject = len(self.file_name_list)
        print('Total number of subjects in fNIRS database:', self.num_subject)

    def read(self, data_index):
        assert 1 <= data_index <= self.num_subject

        file_path = os.path.join(self.root_path, f"sub_{int(data_index)}.csv")
        data = np.loadtxt(file_path, delimiter=",", skiprows=1)
        label = (np.trunc(data[:, -1])).astype(np.int64)
        data = data[:, :-1].astype(np.float32)

        mean = data.mean(axis=0, keepdims=True)
        std = data.std(axis=0, keepdims=True)
        data = (data - mean) / std

        return data.copy(), label.copy()


class SleepDataLoader:
    def __init__(self, path='/data/eeggroup/public_dataset/SleepEdf_Dataset/physionet.org/files/sleep-edfx/1.0.0/'):
        self.root_path = path
        self.npy_path = "/data/eeggroup/public_dataset/SleepEdf_Dataset/physionet.org/npydata/"
        self.SC_tags = ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'Resp oro-nasal', 'EMG submental', 'Temp rectal',
                        'Event marker']
        self.ST_tags = ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'EMG submental', 'Event marker']
        self.SC_Users = [['SC4001E0', 'SC4001EC'], ['SC4002E0', 'SC4002EC'], ['SC4011E0', 'SC4011EH'],
                         ['SC4012E0', 'SC4012EC'], ['SC4021E0', 'SC4021EH'], ['SC4022E0', 'SC4022EJ'],
                         ['SC4031E0', 'SC4031EC'], ['SC4032E0', 'SC4032EP'], ['SC4041E0', 'SC4041EC'],
                         ['SC4042E0', 'SC4042EC'], ['SC4051E0', 'SC4051EC'], ['SC4052E0', 'SC4052EC'],
                         ['SC4061E0', 'SC4061EC'], ['SC4062E0', 'SC4062EC'], ['SC4071E0', 'SC4071EC'],
                         ['SC4072E0', 'SC4072EH'], ['SC4081E0', 'SC4081EC'], ['SC4082E0', 'SC4082EP'],
                         ['SC4091E0', 'SC4091EC'], ['SC4092E0', 'SC4092EC'], ['SC4101E0', 'SC4101EC'],
                         ['SC4102E0', 'SC4102EC'], ['SC4111E0', 'SC4111EC'], ['SC4112E0', 'SC4112EC'],
                         ['SC4121E0', 'SC4121EC'], ['SC4122E0', 'SC4122EV'], ['SC4131E0', 'SC4131EC'],
                         ['SC4141E0', 'SC4141EU'], ['SC4142E0', 'SC4142EU'], ['SC4151E0', 'SC4151EC'],
                         ['SC4152E0', 'SC4152EC'], ['SC4161E0', 'SC4161EC'], ['SC4162E0', 'SC4162EC'],
                         ['SC4171E0', 'SC4171EU'], ['SC4172E0', 'SC4172EC'], ['SC4181E0', 'SC4181EC'],
                         ['SC4182E0', 'SC4182EC'], ['SC4191E0', 'SC4191EP'], ['SC4192E0', 'SC4192EV'],
                         ['SC4201E0', 'SC4201EC'], ['SC4202E0', 'SC4202EC'], ['SC4211E0', 'SC4211EC'],
                         ['SC4212E0', 'SC4212EC'], ['SC4221E0', 'SC4221EJ'], ['SC4222E0', 'SC4222EC'],
                         ['SC4231E0', 'SC4231EJ'], ['SC4232E0', 'SC4232EV'], ['SC4241E0', 'SC4241EC'],
                         ['SC4242E0', 'SC4242EA'], ['SC4251E0', 'SC4251EP'], ['SC4252E0', 'SC4252EU'],
                         ['SC4261F0', 'SC4261FM'], ['SC4262F0', 'SC4262FC'], ['SC4271F0', 'SC4271FC'],
                         ['SC4272F0', 'SC4272FM'], ['SC4281G0', 'SC4281GC'], ['SC4282G0', 'SC4282GC'],
                         ['SC4291G0', 'SC4291GA'], ['SC4292G0', 'SC4292GC'], ['SC4301E0', 'SC4301EC'],
                         ['SC4302E0', 'SC4302EV'], ['SC4311E0', 'SC4311EC'], ['SC4312E0', 'SC4312EM'],
                         ['SC4321E0', 'SC4321EC'], ['SC4322E0', 'SC4322EC'], ['SC4331F0', 'SC4331FV'],
                         ['SC4332F0', 'SC4332FC'], ['SC4341F0', 'SC4341FA'], ['SC4342F0', 'SC4342FA'],
                         ['SC4351F0', 'SC4351FA'], ['SC4352F0', 'SC4352FV'], ['SC4362F0', 'SC4362FC'],
                         ['SC4371F0', 'SC4371FA'], ['SC4372F0', 'SC4372FC'], ['SC4381F0', 'SC4381FC'],
                         ['SC4382F0', 'SC4382FW'], ['SC4401E0', 'SC4401EC'], ['SC4402E0', 'SC4402EW'],
                         ['SC4411E0', 'SC4411EJ'], ['SC4412E0', 'SC4412EM'], ['SC4421E0', 'SC4421EA'],
                         ['SC4422E0', 'SC4422EA'], ['SC4431E0', 'SC4431EM'], ['SC4432E0', 'SC4432EM'],
                         ['SC4441E0', 'SC4441EC'], ['SC4442E0', 'SC4442EV'], ['SC4451F0', 'SC4451FY'],
                         ['SC4452F0', 'SC4452FW'], ['SC4461F0', 'SC4461FA'], ['SC4462F0', 'SC4462FJ'],
                         ['SC4471F0', 'SC4471FA'], ['SC4472F0', 'SC4472FA'], ['SC4481F0', 'SC4481FV'],
                         ['SC4482F0', 'SC4482FJ'], ['SC4491G0', 'SC4491GJ'], ['SC4492G0', 'SC4492GJ'],
                         ['SC4501E0', 'SC4501EW'], ['SC4502E0', 'SC4502EM'], ['SC4511E0', 'SC4511EJ'],
                         ['SC4512E0', 'SC4512EW'], ['SC4522E0', 'SC4522EM'], ['SC4531E0', 'SC4531EM'],
                         ['SC4532E0', 'SC4532EV'], ['SC4541F0', 'SC4541FA'], ['SC4542F0', 'SC4542FW'],
                         ['SC4551F0', 'SC4551FC'], ['SC4552F0', 'SC4552FW'], ['SC4561F0', 'SC4561FJ'],
                         ['SC4562F0', 'SC4562FJ'], ['SC4571F0', 'SC4571FV'], ['SC4572F0', 'SC4572FC'],
                         ['SC4581G0', 'SC4581GM'], ['SC4582G0', 'SC4582GP'], ['SC4591G0', 'SC4591GY'],
                         ['SC4592G0', 'SC4592GY'], ['SC4601E0', 'SC4601EC'], ['SC4602E0', 'SC4602EJ'],
                         ['SC4611E0', 'SC4611EG'], ['SC4612E0', 'SC4612EA'], ['SC4621E0', 'SC4621EV'],
                         ['SC4622E0', 'SC4622EJ'], ['SC4631E0', 'SC4631EM'], ['SC4632E0', 'SC4632EA'],
                         ['SC4641E0', 'SC4641EP'], ['SC4642E0', 'SC4642EP'], ['SC4651E0', 'SC4651EP'],
                         ['SC4652E0', 'SC4652EG'], ['SC4661E0', 'SC4661EJ'], ['SC4662E0', 'SC4662EJ'],
                         ['SC4671G0', 'SC4671GJ'], ['SC4672G0', 'SC4672GV'], ['SC4701E0', 'SC4701EC'],
                         ['SC4702E0', 'SC4702EA'], ['SC4711E0', 'SC4711EC'], ['SC4712E0', 'SC4712EA'],
                         ['SC4721E0', 'SC4721EC'], ['SC4722E0', 'SC4722EM'], ['SC4731E0', 'SC4731EM'],
                         ['SC4732E0', 'SC4732EJ'], ['SC4741E0', 'SC4741EA'], ['SC4742E0', 'SC4742EC'],
                         ['SC4751E0', 'SC4751EC'], ['SC4752E0', 'SC4752EM'], ['SC4761E0', 'SC4761EP'],
                         ['SC4762E0', 'SC4762EG'], ['SC4771G0', 'SC4771GC'], ['SC4772G0', 'SC4772GC'],
                         ['SC4801G0', 'SC4801GC'], ['SC4802G0', 'SC4802GV'], ['SC4811G0', 'SC4811GG'],
                         ['SC4812G0', 'SC4812GV'], ['SC4821G0', 'SC4821GC'], ['SC4822G0', 'SC4822GC'], ]
        self.ST_Users = [['ST7011J0', 'ST7011JP'], ['ST7012J0', 'ST7012JP'], ['ST7021J0', 'ST7021JM'],
                         ['ST7022J0', 'ST7022JM'], ['ST7041J0', 'ST7041JO'], ['ST7042J0', 'ST7042JO'],
                         ['ST7051J0', 'ST7051JA'], ['ST7052J0', 'ST7052JA'], ['ST7061J0', 'ST7061JR'],
                         ['ST7062J0', 'ST7062JR'], ['ST7071J0', 'ST7071JA'], ['ST7072J0', 'ST7072JA'],
                         ['ST7081J0', 'ST7081JW'], ['ST7082J0', 'ST7082JW'], ['ST7091J0', 'ST7091JE'],
                         ['ST7092J0', 'ST7092JE'], ['ST7101J0', 'ST7101JE'], ['ST7102J0', 'ST7102JE'],
                         ['ST7111J0', 'ST7111JE'], ['ST7112J0', 'ST7112JE'], ['ST7121J0', 'ST7121JE'],
                         ['ST7122J0', 'ST7122JE'], ['ST7131J0', 'ST7131JR'], ['ST7132J0', 'ST7132JR'],
                         ['ST7141J0', 'ST7141JE'], ['ST7142J0', 'ST7142JE'], ['ST7151J0', 'ST7151JA'],
                         ['ST7152J0', 'ST7152JA'], ['ST7161J0', 'ST7161JM'], ['ST7162J0', 'ST7162JM'],
                         ['ST7171J0', 'ST7171JA'], ['ST7172J0', 'ST7172JA'], ['ST7181J0', 'ST7181JR'],
                         ['ST7182J0', 'ST7182JR'], ['ST7191J0', 'ST7191JR'], ['ST7192J0', 'ST7192JR'],
                         ['ST7201J0', 'ST7201JO'], ['ST7202J0', 'ST7202JO'], ['ST7211J0', 'ST7211JJ'],
                         ['ST7212J0', 'ST7212JJ'], ['ST7221J0', 'ST7221JA'], ['ST7222J0', 'ST7222JA'],
                         ['ST7241J0', 'ST7241JO'], ['ST7242J0', 'ST7242JO'], ]
        self.dir_dict = dict()
        self.dir_dict = {'sleep-cassette': self.SC_Users, 'sleep-telemetry': self.ST_Users}
        self.label_dict = {
            'Sleep stage W': 0,
            'Sleep stage 1': 1,
            'Sleep stage 2': 2,
            'Sleep stage 3': 3,
            'Sleep stage 4': 3,
            'Sleep stage R': 4,
        }
        self.channels = ['EEG Fpz-Cz', 'EOG horizontal']
        self.sample_rate = 100
        self.num_subject = len(self.SC_Users)
        print('Total number of subjects in Sleep-Cassette database:', self.num_subject)

    def store(self, data, label, study_type="sleep-cassette", data_index=0):
        data_npy_path = os.path.join(self.npy_path, study_type, self.dir_dict[study_type][data_index][0][:6] +
                                     "_data.npy")
        label_npy_path = os.path.join(self.npy_path, study_type, self.dir_dict[study_type][data_index][0][:6] +
                                      "_label.npy")
        np.save(data_npy_path, data)
        np.save(label_npy_path, label)
        print("npy data store!")

    def read(self, study_type="sleep-cassette", data_index=0):
        assert 0 <= data_index <= self.num_subject - 1

        data_npy_path = os.path.join(self.npy_path, study_type, self.dir_dict[study_type][data_index][0][:6] +
                                     "_data.npy")
        label_npy_path = os.path.join(self.npy_path, study_type, self.dir_dict[study_type][data_index][0][:6] +
                                      "_label.npy")
        if os.path.exists(data_npy_path) and os.path.exists(label_npy_path):
            epoch_list = np.load(data_npy_path).astype(np.float32)
            label_list = np.load(label_npy_path).astype(np.int64)
            print("npy data read!")
            return epoch_list.copy(), label_list.copy()

        epoch_list = []
        label_list = []
        raw = read_raw_edf(os.path.join(self.root_path, study_type, self.dir_dict[study_type][data_index][0] +
                                        "-PSG.edf"), preload=True)
        raw.pick_channels(self.channels)
        raw.filter(0.3, 35, fir_design='firwin')
        annotation = mne.read_annotations(os.path.join(self.root_path, study_type,
                                                       self.dir_dict[study_type][data_index][1] + "-Hypnogram.edf"))
        raw.set_annotations(annotation, emit_warning=False)

        events_train, event_id = mne.events_from_annotations(raw, chunk_duration=30.)
        if 'Sleep stage ?' in event_id.keys():
            event_id.pop('Sleep stage ?')
        if 'Movement time' in event_id.keys():
            event_id.pop('Movement time')
        tmax = 30. - 1. / raw.info['sfreq']
        assert self.sample_rate == raw.info['sfreq']
        epoch_train = mne.Epochs(raw=raw, events=events_train,
                                 event_id=event_id, tmin=0., tmax=tmax, baseline=None)

        labels = []
        for epoch_annotation in epoch_train.get_annotations_per_epoch():
            labels.append(epoch_annotation[0][2])

        length = len(labels)
        a = []
        for i, label in enumerate(labels):
            if label != 'Sleep stage W':
                a.append(i)
        print(len(a))

        start = max(a[0] - 60, 0)
        end = min(a[-1] + 60, length)
        print("start time : ", start, " end time : ", end)
        epochs = epoch_train[start:end]
        labels_ = labels[start:end]
        print(epochs)
        for epoch in epochs:
            epoch_list.append(epoch)
        for label in labels_:
            label_list.append(self.label_dict[label])

        # epoch_num -> (epoch_num * 3000)
        label_list = np.expand_dims(np.array(label_list), axis=-1)
        label_list = label_list.repeat(epoch_list[0].shape[-1], axis=-1).reshape(-1).astype(np.int64)
        # epoch_num x 2 x 3000 -> (epoch_num * 3000) x 2
        epoch_list = np.array(epoch_list).transpose(0, 2, 1).reshape(-1, len(self.channels)).astype(np.float32)

        mean = epoch_list.mean(axis=0, keepdims=True)
        std = epoch_list.std(axis=0, keepdims=True)
        epoch_list = (epoch_list - mean) / std

        self.store(epoch_list, label_list, study_type, data_index)
        return epoch_list.copy(), label_list.copy()


class HHARDataLoader:
    def __init__(self, path='/data/eeggroup/public_dataset/hhar'):
        self.root_path = path
        self.label = [0, 1, 2, 3, 4, 5]
        self.devices = ['0', '1', '2']
        self.num_group = len(self.devices)
        self.sample_rate = 50
        self.threshold_t = 10000
        self.label_dict = {'stand': 0,
                           'sit': 1,
                           'walk': 2,
                           'bike': 3,
                           'stairsup': 4,
                           'stairsdown': 5,
                           'null': 6}
        self.device_env_mapping = {'nexus4_1': 'nexus4',
                                   'nexus4_2': 'nexus4',
                                   's3_1': 's3',
                                   's3_2': 's3',
                                   's3mini_1': 's3mini',
                                   's3mini_2': 's3mini',
                                   'gear_1': 'gear',
                                   'gear_2': 'gear',
                                   'lgwatch_1': 'lgwatch',
                                   'lgwatch_2': 'lgwatch'}

    def read_data(self, ids_idx):
        data_filename = f'data_{ids_idx}.npy'
        data_path = os.path.join(self.root_path, 'grouped_data_labels', data_filename)

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"The file {data_filename} does not exist in.")
        data = np.load(data_path)

        return data

    def read_label(self, ids_idx):
        labels_filename = f'labels_{ids_idx}.npy'
        labels_path = os.path.join(self.root_path, 'grouped_data_labels', labels_filename)

        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"The file {labels_filename} does not exist.")
        labels = np.load(labels_path)

        return labels

    def load_and_preprocess(self):
        threshold_t = self.threshold_t
        file_path = self.root_path

        label_dict = self.label_dict
        device_env_mapping = self.device_env_mapping

        ## Fetch all data and put it all in a big dict
        data_dict = {}
        for file in glob.glob(os.path.join(file_path, '*.csv')):
            # Get modality
            if 'gyroscope' in file:
                mod = 'gyro'
            elif 'accelerometer' in file:
                mod = 'acc'

            # Get number of time steps for all recordings
            with open(file) as f:
                data = csv.reader(f)
                next(data)
                for row in data:
                    if row[8] not in data_dict.keys():
                        data_dict[row[8]] = {}
                    if row[6] not in data_dict[row[8]].keys():
                        data_dict[row[8]][row[6]] = {}
                    if mod not in data_dict[row[8]][row[6]].keys():
                        data_dict[row[8]][row[6]][mod] = {}
                        data_dict[row[8]][row[6]][mod]['n_pt'] = 0

                    data_dict[row[8]][row[6]][mod]['n_pt'] += 1

            # Get data
            with open(file) as f:
                data = csv.reader(f)
                next(data)
                for row in data:
                    if 'index' not in data_dict[row[8]][row[6]][mod].keys():
                        i = 0
                        data_dict[row[8]][row[6]][mod]['index'] = np.zeros((data_dict[row[8]][row[6]][mod]['n_pt']))
                        data_dict[row[8]][row[6]][mod]['time'] = np.zeros((data_dict[row[8]][row[6]][mod]['n_pt']))
                        data_dict[row[8]][row[6]][mod]['meas'] = np.zeros((data_dict[row[8]][row[6]][mod]['n_pt'], 3),
                                                                          dtype=np.float64)
                        data_dict[row[8]][row[6]][mod]['label'] = np.zeros((data_dict[row[8]][row[6]][mod]['n_pt']))

                    data_dict[row[8]][row[6]][mod]['index'][i] = int(row[0])
                    data_dict[row[8]][row[6]][mod]['time'][i] = float(row[2]) / 1e6  # Convert to milliseconds
                    data_dict[row[8]][row[6]][mod]['meas'][i, :] = [float(row[3]), float(row[4]), float(row[5])]
                    data_dict[row[8]][row[6]][mod]['label'][i] = int(label_dict[row[9]])

                    i += 1

        to_delete = []
        for device in data_dict.keys():
            for sub in data_dict[device].keys():
                if len(data_dict[device][sub].keys()) != 2:
                    to_delete.append((device, sub))
                    continue
                for mod in data_dict[device][sub].keys():
                    if data_dict[device][sub][mod]['n_pt'] < threshold_t:
                        to_delete.append((device, sub))
                        break
        for key in to_delete:
            del data_dict[key[0]][key[1]]

        for device in data_dict.keys():
            for sub in data_dict[device].keys():
                for mod in data_dict[device][sub].keys():
                    # Sort by index
                    index_sort = np.argsort(data_dict[device][sub][mod]['index'])
                    data_dict[device][sub][mod]['index'] = np.take_along_axis(data_dict[device][sub][mod]['index'],
                                                                              index_sort, axis=0)
                    data_dict[device][sub][mod]['time'] = np.take_along_axis(data_dict[device][sub][mod]['time'],
                                                                             index_sort, axis=0)
                    data_dict[device][sub][mod]['meas'] = data_dict[device][sub][mod]['meas'][index_sort, :]
                    data_dict[device][sub][mod]['label'] = np.take_along_axis(data_dict[device][sub][mod]['label'],
                                                                              index_sort, axis=0)

                    # This is to take data that is within recording time
                    inliers = np.argwhere(
                        np.logical_and(data_dict[device][sub][mod]['time'][0] <= data_dict[device][sub][mod]['time'],
                                       data_dict[device][sub][mod]['time'] <= data_dict[device][sub][mod]['time'][-1]))[
                              :, 0]

                    # Sort by time value
                    time_sort = np.argsort(data_dict[device][sub][mod]['time'][inliers])

                    data_dict[device][sub][mod]['index'] = data_dict[device][sub][mod]['index'][inliers][time_sort]
                    data_dict[device][sub][mod]['time'] = data_dict[device][sub][mod]['time'][inliers][time_sort]
                    data_dict[device][sub][mod]['meas'] = data_dict[device][sub][mod]['meas'][inliers][time_sort, :]
                    data_dict[device][sub][mod]['label'] = data_dict[device][sub][mod]['label'][inliers][time_sort]

        for device in data_dict.keys():
            for i, sub in enumerate(data_dict[device].keys()):

                tmin = np.max([data_dict[device][sub]['gyro']['time'][0], data_dict[device][sub]['acc']['time'][0]])
                tmax = np.min([data_dict[device][sub]['gyro']['time'][-1], data_dict[device][sub]['acc']['time'][-1]])

                gyro_in = np.argwhere(np.logical_and(tmin <= data_dict[device][sub]['gyro']['time'],
                                                     data_dict[device][sub]['gyro']['time'] <= tmax))[:, 0]
                acc_in = np.argwhere(np.logical_and(tmin <= data_dict[device][sub]['acc']['time'],
                                                    data_dict[device][sub]['acc']['time'] <= tmax))[:, 0]

                gyro_in = gyro_in[data_dict[device][sub]['gyro']['label'][gyro_in] != 6]
                acc_in = acc_in[data_dict[device][sub]['acc']['label'][acc_in] != 6]

                # scale
                data_dict[device][sub]['gyro']['meas'] = scale(data_dict[device][sub]['gyro']['meas'])
                data_dict[device][sub]['acc']['meas'] = scale(data_dict[device][sub]['acc']['meas'])

                gyro_time = data_dict[device][sub]['gyro']['time'][gyro_in]
                acc_time = data_dict[device][sub]['acc']['time'][acc_in]
                gyro_meas = data_dict[device][sub]['gyro']['meas'][gyro_in]
                acc_meas = data_dict[device][sub]['acc']['meas'][acc_in]
                gyro_labels = data_dict[device][sub]['gyro']['label'][gyro_in]
                acc_labels = data_dict[device][sub]['acc']['label'][acc_in]

                aligned_data = []
                aligned_labels = []

                gyro_idx, acc_idx = 0, 0
                while gyro_idx < len(gyro_time) and acc_idx < len(acc_time):
                    if abs(gyro_time[gyro_idx] - acc_time[acc_idx]) <= 50:
                        if gyro_labels[gyro_idx] == acc_labels[acc_idx]:
                            combined_measurement = np.concatenate((gyro_meas[gyro_idx], acc_meas[acc_idx]))
                            aligned_data.append(combined_measurement)
                            aligned_labels.append(gyro_labels[gyro_idx])

                        gyro_idx += 1
                        acc_idx += 1
                    elif gyro_time[gyro_idx] < acc_time[acc_idx]:
                        gyro_idx += 1
                    else:
                        acc_idx += 1

                data = np.array(aligned_data)
                labels = np.array(aligned_labels).reshape(-1, 1)

                env = device_env_mapping[device]

                with h5py.File(os.path.join(file_path, 'HHAR.h5'), 'a') as hf:
                    if env not in hf.keys():
                        g = hf.create_group(env)
                        g.create_dataset('data', data=data.astype('float32'), dtype='float32', maxshape=(None, 6))
                        g.create_dataset('labels', data=labels.astype('float32'), dtype='int_', maxshape=(None, 1))
                    else:
                        hf[env]['data'].resize((hf[env]['data'].shape[0] + data.shape[0]), axis=0)
                        hf[env]['data'][-data.shape[0]:, :] = data
                        hf[env]['labels'].resize((hf[env]['labels'].shape[0] + labels.shape[0]), axis=0)
                        hf[env]['labels'][-labels.shape[0]:, :] = labels

    def regroup(self):
        file_path = os.path.join(self.root_path, 'HHAR.h5')
        output_dir = os.path.join(self.root_path, 'grouped_data_labels')

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        all_data = []
        all_labels = []
        merge_data = []
        merge_labels = []

        with h5py.File(file_path, 'r') as file:
            for group_name in file:
                group = file[group_name]
                data = group['data'][:]
                labels = group['labels'][:]

                if group_name in ['lgwatch', 'gear', 's3mini']:
                    merge_data.append(data)
                    merge_labels.append(labels)
                else:
                    all_data.append(data)
                    all_labels.append(labels)

        if merge_data and merge_labels:
            merged_data = np.concatenate(merge_data, axis=0)
            merged_labels = np.concatenate(merge_labels, axis=0)

            all_data.append(merged_data)
            all_labels.append(merged_labels)

        for i, data in enumerate(all_data):
            np.save(os.path.join(output_dir, f'data_{i}.npy'), data.astype('float32'))

        for i, labels in enumerate(all_labels):
            np.save(os.path.join(output_dir, f'labels_{i}.npy'), labels.astype('int64'))

    def read(self, ids_idx):
        assert 0 <= ids_idx <= self.num_group - 1

        data_filename = f'data_{ids_idx}.npy'
        data_path = os.path.join(self.root_path, 'grouped_data_labels', data_filename)
        if not os.path.exists(data_path):
            self.load_and_preprocess()
            self.regroup()

        data = self.read_data(ids_idx)
        label = self.read_label(ids_idx)

        return data.copy(), label.reshape(-1).copy()
