import argparse
import sys
import os
import glob
from utils.str2bool import str2bool

def arg_parser(role):
    """
    Parse arguments from command line and from config file (path specified in command line) and return them
    :param role (str): either 'train', 'eval' or 'demo': to get the right list of parameters depending of the application
    :return: dict of arguments
    """
    parser = argparse.ArgumentParser(description='VGG_SSD/MobileNet2_SSD/YOLO3 Detector Training/Inference With Pytorch')

    parser.add_argument('--image_set', default=None, help='[KAIST] Imageset')
    parser.add_argument("--data_config_path", type=str, default=None, help="path to data config file")
    parser.add_argument("--model", type=str, default=None, help="Model to test, either 'VGG_SSD','MOBILENET2_SSD' or 'YOLO'")
    parser.add_argument('--image_fusion', default=-1, type=int,
                        help='[KAIST]: type of image fusion: [0: visible], [1: lwir] [2: inverted LWIR] [3: early fused]')

    if role == "train":
        parser.add_argument('--resume', default=None, type=str, help='Checkpoint state_dict file to resume training from')
        parser.add_argument('--start_iter', default=0, type=int, help='Resume training at this iter')
        parser.add_argument('--visdom', default=False, type=str2bool, help='Use visdom for loss visualization')
        parser.add_argument('--save_folder', default='checkpoints', help='Directory for saving checkpoint models')
        parser.add_argument('--show_dataset', default=False, type=str2bool, help='Show every image used ?')

    elif role in {"eval", "demo"}:
        parser.add_argument('--trained_model', default=None, type=str, help='Trained state_dict file path to open')
        parser.add_argument('--corrected_annotations', default=False, type=str2bool,
                            help='[KAIST] do we use the corrected annotations ? (must ahve compatible imageset (VPY-test-strict-type-5)')
    else:
        raise NotImplementedError


    # command line + file args as 2 dictionnaries
    args_cmd = vars(parser.parse_args())
    args_cmd = {key: value for key, value in args_cmd.items() if value is not None} # remove non attributed values from parser
    args_file = parse_data_config(args_cmd['data_config_path'])

    # remove unused args (other model,...)
    if args_cmd["model"] == "VGG_SSD":
        args_file = {key.replace("vgg_", ""): value for key, value in args_file.items() if (not key.startswith("mobilenet")) and (not key.startswith("yolo"))}
    elif args_cmd["model"] == "MOBILENET2_SSD":
        args_file = {key.replace("mobilenet_", ""): value for key, value in args_file.items() if (not key.startswith("vgg")) and (not key.startswith("yolo"))}
    elif args_cmd["model"] == "YOLO":
        args_file = {key: value for key, value in args_file.items() if (not key.startswith("vgg")) and (not key.startswith("mobilenet"))}
    else:
        raise NotImplementedError

    # remove duplicates entries, command line has priority on config file
    duplicates = (args_file.keys() & args_cmd.keys())
    for key in duplicates:
        del args_file[key]

    # fuse two dictionnaries
    args = {**args_cmd, **args_file}


    return args

def check_args(args, role):
    if role == "train":
        if args['save_frequency'] < 0:
            print("save frequency must be > 0")
            sys.exit(-1)

        #prepare output folder:
        # - create it if required
        # - warn if not empty
        # - do not warn and remove content if name contains "tmp"
        if not os.path.exists(args['save_folder']):
            os.makedirs(args['save_folder'])
        if len(os.listdir(args['save_folder'])) != 0:
            print("Save directory is not empty! : {}".format(args['save_folder']))
            if "tmp" not in args['save_folder']:
                print("not a tmp folder, you must fix it yourself!")
                sys.exit(-1)
            else:
                print("but as the save folder contains 'tmp', we remove old data:")
                files = glob.glob(args['save_folder'] + '/*')
                for f in files:
                    if f.endswith('.weights') or f.endswith('.pth'):
                        print("rm {}".format(f))
                        os.remove(f)

    # KAIST needs image_fusion
    if args['name'] == "KAIST":
        if args['image_fusion'] == -1:
            print("image fusion must be specified")
            sys.exit(-1)

    # Need existing dataset_root
    if not os.path.exists(args['dataset_root']):
        print('Must specify *existing* dataset_root')
        sys.exit(-1)

    # Warn about other than KAIST datasets
    if args["name"] != "KAIST":
        print("Experimental support, not tested!")

    # SSD-based models supports:
    # - KAIST
    # - VOC (to be tested)
    # YOLO-based models supports:
    # - KAIST
    # - COCO
    if args["model"] in {"VGG_SSD", "MOBILENET2_SSD"}:
        if args["name"] not in {"VOC", "KAIST"}:
            print("Dataset {} not supported with model {}".format(args["name"], args["model"]))
            sys.exit(-1)
    elif args["model"] == "YOLO":
        if args["name"] not in {"COCO", "KAIST"}:
            print("Dataset {} not supported with model {}".format(args["name"], args["model"]))
            sys.exit(-1)
    else:
        print("Model {} is not supported".format(args["model"]))
        sys.exit(-1)
    return

def parse_yolo_model_config(path):
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines] # get rid of fringe whitespaces
    module_defs = []
    for line in lines:
        if line.startswith('['): # This marks the start of a new block
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()
    return module_defs

def parse_data_config(path):
    """
    Parses the data configuration file
    Format of parameters:
    param_name = param_value
    beware: param_value is evaluated
    see config/kaist.cfg as examples
    comments start by: '#' or '['
    :param path: path of the config file
    :return: list of parameters from file
    """
    options = dict()
    with open(path, 'r') as fp:
        lines = fp.readlines()
        for line in lines:
            line = line.split("#")[0]
            line = line.strip()
            if line == '' or line.startswith('#') or line.startswith('['):
                continue
            key, value = line.split('=')
            try:
                value = eval(value)
            except:
                value = value
            options[key.strip()] = value.strip() if type(value)==str else value
    return options