#!/usr/bin/env python
# coding: utf-8
import yaml
import torch
import argparse
import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
##NEED TO SET
##model= nn.DataParallel(model,device_ids = [0, 1])
# For reproducibility, comment these may speed up training
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
##HOME DIRECTORY
CONFIG = "/disk/scratch2/s1834237/MLP_MULTIPLE_GPU/MLP_Group_Project/tedlium_model/config/ted/asr_example.yaml"
#CONFIG = "/home/wassim_jabrane/MLP_Group_Project/tedlium_model/config/ted/asr_example.yaml"
NAME = "tedlium3"
# LOG_DIR = "/home/szy/Documents/code/espnet/egs/tedlium3/asr1/tedlium/log/"
# CHECK_POINT_DIR = "/home/szy/Documents/code/espnet/egs/tedlium3/asr1/tedlium/check_point/"

# Arguments
parser = argparse.ArgumentParser(description='Training E2E asr.')
parser.add_argument('--config', type=str, help='Path to experiment config.')
parser.add_argument('--name', default=None, type=str, help='Name for logging.')
parser.add_argument('--logdir', default='log/', type=str,help='Logging path.', required=False)
parser.add_argument('--ckpdir', default='ckpt/', type=str,help='Checkpoint path.', required=False)
parser.add_argument('--outdir', default='result/', type=str,help='Decode output path.', required=False)
parser.add_argument('--load', default=None, type=str,help='Load pre-trained model (for training only)', required=False)
parser.add_argument('--seed', default=0, type=int,help='Random seed for reproducable results.', required=False)
parser.add_argument('--cudnn-ctc', action='store_true',help='Switches CTC backend from torch to cudnn')
parser.add_argument('--njobs', default=1, type=int,help='Number of threads for dataloader/decoding.', required=False)
parser.add_argument('--cpu', action='store_true', help='Disable GPU training.')
parser.add_argument('--no-pin', action='store_true',help='Disable pin-memory for dataloader')
parser.add_argument('--test', action='store_true', help='Test the model.')
parser.add_argument('--no-msg', action='store_true', help='Hide all messages.')
parser.add_argument('--lm', action='store_true',help='Option for training RNNLM.')
# Following features in development.
parser.add_argument('--amp', action='store_true', help='Option to enable AMP.')
parser.add_argument('--reserve-gpu', default=0, type=float,help='Option to reserve GPU ram for training.')
parser.add_argument('--jit', action='store_true',help='Option for enabling jit in pytorch. (feature in development)')
###

script = ['--config', CONFIG, '--name', NAME]
paras = parser.parse_args(script)
setattr(paras, 'gpu', not paras.cpu)
setattr(paras, 'pin_memory', not paras.no_pin)
setattr(paras, 'verbose', not paras.no_msg)
config = yaml.load(open(paras.config, 'r'), Loader=yaml.FullLoader)

np.random.seed(paras.seed)
torch.manual_seed(paras.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(paras.seed)

devices = 'cuda:0,1'

# Hack to preserve GPU ram just incase OOM later on server
#if paras.gpu and paras.reserve_gpu > 0:
#    buff = torch.randn(int(paras.reserve_gpu*1e9//4)).cuda()
#    del buff

##CHECK WE ONLY SEE THE GIVEN DEVICES##
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")

if paras.lm:
    # Train RNNLM
    from bin.train_lm import Solver
    mode = 'train'
else:
    if paras.test:
        # Test ASR
        assert paras.load is None, 'Load option is mutually exclusive to --test'
        from bin.test_asr import Solver
        mode = 'test'
    else:
        # Train ASR
        from bin.train_asr import Solver
        mode = 'train'


solver = Solver(config, paras, mode)
solver.load_data()
solver.set_model()
solver.exec()
