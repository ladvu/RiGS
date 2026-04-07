
import sys
import argparse
import torch
import json
from os.path import dirname, join
RAFT_PATH_ROOT = join(dirname(__file__), 'RAFT')
RAFT_PATH_CORE = join(RAFT_PATH_ROOT, 'core')
sys.path.append(RAFT_PATH_CORE)
from raft import RAFT, RAFT2  # nopep8
from core.utils.utils import InputPadder  # nopep8

# %%
# utility functions

def json_to_args(json_path):
    # return a argparse.Namespace object
    with open(json_path, 'r') as f:
        data = json.load(f)
    args = argparse.Namespace()
    args_dict = args.__dict__
    for key, value in data.items():
        args_dict[key] = value
    return args

def parse_args(parser):
    entry = parser.parse_args(args=[])
    json_path = entry.cfg
    args = json_to_args(json_path)
    args_dict = args.__dict__
    for index, (key, value) in enumerate(vars(entry).items()):
        args_dict[key] = value
    return args

def get_input_padder(shape):
    return InputPadder(shape, mode='sintel')


def load_RAFT(model_path, cfg_path):
    args = json_to_args(cfg_path)
    net = RAFT2(args)
    state_dict = torch.load(model_path)
    print('Loaded pretrained RAFT model from', model_path)
    new_state_dict = {}
    for k in state_dict:
        if 'module' in k:
            name = k[7:]
        else:
            name = k
        new_state_dict[name] = state_dict[k]
    net.load_state_dict(new_state_dict)
    return net.eval()

if __name__ == "__main__":
    net = load_RAFT(model_path='third_party/RAFT/models/Tartan-C-T432x960-M.pth')
    print(net)