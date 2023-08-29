import argparse, datetime
from utils.utils_ import *
from argparse import Namespace
import os

# parser：
parser = argparse.ArgumentParser(description='general')

# str
parser.add_argument('--model_dir', default='Saved_Models/', help='folder to output model checkpoints')
parser.add_argument('--id', default='XXXX', help='name')
parser.add_argument('--fn_model_type', default='cnn', help='surrogate model. Choose either `fnn` or `cnn`')
parser.add_argument('--teacher_model_name', default='resnet18', help='teacher model. Choose either `resnet18` or `resnet50`')
parser.add_argument('--dataset', required=True, default='cifar10', help="dataset. `mnist`, `cifar10`, and `imagenet` are available. For ImageNet or other dataset, please modify the path in the code directly.")
parser.add_argument('--save_path', default="./data", help="path to save poisoned data")
parser.add_argument('--loss', default="KL", help="loss type")
parser.add_argument('--norm_type', default="np.inf", help="norm type. Choose either `np.inf` or `2`")
parser.add_argument('--cuda_visible_devices', default="2", help='''specify which GPU to run an application on,
                                                                example: '0,' or "0,1" ''')

# int
parser.add_argument('--val_size', type=int, default=10000, help="size of validation data")
parser.add_argument('--t', type=int, default=64, help="time step used to compute poisoned data")
parser.add_argument('--nb_iter', type=int, default=10, help="number of iteration used to generate poisoned data")
parser.add_argument('--block_size', type=int, default=512, help="block size of B-NTGA")
parser.add_argument('--batch_size', type=int, default=128, help="batch size, refer to batch size of test set when making poison data")
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')

# float
parser.add_argument('--eps', type=float, help="epsilon. Strength of NTGA")
parser.add_argument('--T', type=float, default=4.0, help="temperature of softmax in KLdiv loss")
parser.add_argument('--sparse_ratio', type=float, default=0.2, help="sparse ratio of the logits of stingy teacher")
parser.add_argument('--step_size', type=float, default=1.1, help="step size to calculate the args.eps_iter")

# bool
parser.add_argument('--augmentation', action='store_true', help="whether to use data augmentation") #not use so far


def get_args(ipynb=False):
    if ipynb:  # for jupyter so that default args are passed
        args, remaining = parser.parse_known_args([])
    else:
        args, remaining = parser.parse_known_args()

    printc.yellow('Parsed options:\n{}'.format(vars(args)))

    # show in txt file(data name):
    txt = []
    current_time = datetime.datetime.now()
    current_time = current_time.strftime("%m%d-%H_%M")
    txt += ['id:' + str(args.id)]
    txt += ['T:' + current_time]
    txt += ['fn_model_type:' + str(args.fn_model_type)]
    txt += ['t_model_name:' + str(args.teacher_model_name)]
    txt += ['eps:' + str(args.eps)]
    txt += ['nb_iter:' + str(args.nb_iter)]
    txt += ['batch_size:' + str(args.batch_size)]
    txt += ['dataset:' + str(args.dataset)]
    txt += ['val_size:' + str(args.val_size)]
    txt += ['seed:' + str(args.seed)]
    txt += ['t:' + str(args.t)]
    
    model_name = '_'.join([str(c) for c in txt])

    ## In windows recommend to replace ':' with '='
    model_name = model_name.replace(':', '=')
    args.save_name = model_name

    if model_name in [getbase(c) for c in glob(pjoin(args.model_dir, '*'))]:
        printc.red('WARNING: MODEL', model_name, '\nALREADY EXISTS')
    else:
        os.makedirs(pjoin(args.model_dir, model_name))

    return args

def clean_args(args) -> list:
    # for args.model:
    args.model = args.model.replace("-", " ")
    args.model = args.model.split("+")
    return args

def combine_args(*args):
    """assert *args have no same keys(attributes)"""
    args_dict = {}
    for arg in args:
        arg_vars = vars(arg)
        for key in arg_vars:
            if key in args_dict and args_dict[key] != arg_vars[key]:
                raise ValueError(f'Duplicate argument key found: {key} for {args_dict[key]} and {arg_vars[key]}')
            else:
                args_dict[key] = arg_vars[key]
    args_all = Namespace(**args_dict)
    return args_all
