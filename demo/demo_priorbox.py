import torch
from layers import *
from config.parse_config import *
import argparse

def arg_parser():
    parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training With Pytorch')
    parser.add_argument("--data_config_path", type=str, default=None, help="path to data config file")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = vars(arg_parser())
    config = parse_data_config(args['data_config_path'])

    args = {**args, **config}
    del config

    # prepare environnement
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    priorbox = PriorBox(args)
    with torch.no_grad():
        priors = priorbox.demo(mode="vert_only")
        # priors = priorbox.demo(mode="vert_med")
        # priors = priorbox.demo(mode="all")
