import random
from misc import set_seed


window_time_dict = {
    'fNIRS_2': 25,
    'Sleep': 2.5,
    'HHAR': 4,
}

slide_time_dict = {
    'fNIRS_2': 5,
    'Sleep': 1.25,
    'HHAR': 2,
}

d_model_dict = {
    'fNIRS_2': 128,
    'Sleep': 128,
    'HHAR': 128,
}

d_inner_dict = {
    'fNIRS_2': 256,
    'Sleep': 256,
    'HHAR': 256,
}

kernel_size_list_dict = {
    'fNIRS_2': [3, 3],
    'Sleep': [4, 4],
    'HHAR': [4, 4],
}

stride_size_list_dict = {
    'fNIRS_2': [1, 1],
    'Sleep': [2, 2],
    'HHAR': [2, 2],
}

padding_size_list_dict = {
    'fNIRS_2': [1, 1],
    'Sleep': [1, 1],
    'HHAR': [1, 1],
}

down_sampling_dict = {
    'fNIRS_2': 1,
    'Sleep': 4,
    'HHAR': 4,
}

global_group_dict = {
    'fNIRS_2': {
        'g1': [0],
        'g2': [1],
        'g3': [2],
        'g4': [3],
    },
}


def get_exp_dict(data_name):
    if data_name in ['Sleep', 'HHAR']:
        # Only have three groups
        exp_dict = {
            1: [[0], [1], [2]],
            2: [[0], [2], [1]],
            3: [[1], [0], [2]],
            4: [[1], [2], [0]],
            5: [[2], [0], [1]],
            6: [[2], [1], [0]],
        }
    else:
        group_dict = global_group_dict[data_name]
        exp_dict = {
            1: [group_dict['g1'] + group_dict['g2'], group_dict['g3'], group_dict['g4']],
            2: [group_dict['g1'] + group_dict['g2'], group_dict['g4'], group_dict['g3']],
            3: [group_dict['g1'] + group_dict['g3'], group_dict['g2'], group_dict['g4']],
            4: [group_dict['g1'] + group_dict['g3'], group_dict['g4'], group_dict['g2']],
            5: [group_dict['g1'] + group_dict['g4'], group_dict['g2'], group_dict['g3']],
            6: [group_dict['g1'] + group_dict['g4'], group_dict['g3'], group_dict['g2']],
            7: [group_dict['g2'] + group_dict['g3'], group_dict['g1'], group_dict['g4']],
            8: [group_dict['g2'] + group_dict['g3'], group_dict['g4'], group_dict['g1']],
            9: [group_dict['g2'] + group_dict['g4'], group_dict['g1'], group_dict['g3']],
            10: [group_dict['g2'] + group_dict['g4'], group_dict['g3'], group_dict['g1']],
            11: [group_dict['g3'] + group_dict['g4'], group_dict['g1'], group_dict['g2']],
            12: [group_dict['g3'] + group_dict['g4'], group_dict['g2'], group_dict['g1']],
        }
    return exp_dict


def get_choice_default_config(args):
    if args.random_seed is None:
        args.random_seed = random.randint(0, 2 ** 31)
    set_seed(args.random_seed)

    args.window_time = window_time_dict[args.data_name]
    args.slide_time = slide_time_dict[args.data_name]
    args.d_model = d_model_dict[args.data_name]
    args.d_inner = d_inner_dict[args.data_name]
    args.kernel_size_list = kernel_size_list_dict[args.data_name]
    args.stride_size_list = stride_size_list_dict[args.data_name]
    args.padding_size_list = padding_size_list_dict[args.data_name]
    args.down_sampling = down_sampling_dict[args.data_name]

    exp_dict = get_exp_dict(args.data_name)
    exp_patient_list = exp_dict[args.exp_id]
    args.train_patient_list = exp_patient_list[0]
    args.valid_patient_list = exp_patient_list[1]
    args.test_patient_list = exp_patient_list[2]

    config = CLConfig(
        d_model=args.d_model,
        n_head=args.n_head,
        d_inner=args.d_inner,
        dropout=args.dropout,
        kernel_size_list=args.kernel_size_list,
        stride_size_list=args.stride_size_list,
        padding_size_list=args.padding_size_list,
        down_sampling=args.down_sampling,
    )
    return args, config


def set_default_config(parser):
    group_database = parser.add_argument_group('Database')
    group_database.add_argument('--window_time', type=float, default=1,
                                help='The seconds of every sample segment.')
    group_database.add_argument('--slide_time', type=float, default=0.5,
                                help='The sliding seconds between two sample segments.')
    group_database.add_argument('--num_level', type=int, default=5,
                                help='The number of levels.')
    group_database.add_argument('--n_process_loader', type=int, default=50,
                                help='Number of processes to call to load the dataset.')

    group_model = parser.add_argument_group('Architecture')
    group_model.add_argument('--d_model', type=int, default=128,
                             help='The representation dimension of the model.')
    group_model.add_argument('--d_inner', type=int, default=256,
                             help='The dimension of hidden size of the MLP.')
    # For CL model
    group_model.add_argument('--random_seed', type=int, default=None,
                             help="Set a specific random seed.")
    group_model.add_argument('--kernel_size_list', nargs='*', type=int, default=[4, 4],
                             help='The kernel size list of CNN.')
    group_model.add_argument('--stride_size_list', nargs='*', type=int, default=[2, 2],
                             help='The stride size list of CNN.')
    group_model.add_argument('--padding_size_list', nargs='*', type=int, default=[1, 1],
                             help='The padding size list of CNN.')
    group_model.add_argument('--down_sampling', type=int, default=4,
                             help='The down sampling of CNN.')

    group_model.add_argument('--warm_epoch_num', type=int, default=10,
                             help='Epoch number for total iterations in warm-up stage.')
    group_model.add_argument('--cl_epoch_num', type=int, default=30,
                             help='Epoch number for total iterations in CL stage.')
    group_model.add_argument('--level_gap_epoch_num', type=int, default=5,
                             help='Epoch number for adding new level set in CL stage.')
    # For Transformer model
    group_model.add_argument('--n_head', type=int, default=8,
                             help='The number of heads.')
    group_model.add_argument('--dropout', type=float, default=0.1,
                             help='The dropout rate for the Transformer model.')

    return parser


class CLConfig:
    def __init__(
        self,
        d_model,
        d_inner,
        n_head,
        dropout,
        kernel_size_list,
        stride_size_list,
        padding_size_list,
        down_sampling,
        hidden_act='gelu',
        chunk_size_feed_forward=0,
        layer_norm_eps=1e-12,
    ):
        self.d_model = d_model
        self.d_inner = d_inner
        self.n_head = n_head
        self.dropout = dropout
        self.kernel_size_list = kernel_size_list
        self.stride_size_list = stride_size_list
        self.padding_size_list = padding_size_list
        self.down_sampling = down_sampling
        self.hidden_act = hidden_act
        self.chunk_size_feed_forward = chunk_size_feed_forward
        self.layer_norm_eps = layer_norm_eps
