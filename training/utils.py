# imports
import os
import json
import torch
import copy
import math
from colorama import Fore
from torch import autograd
import torch.nn as nn


########################################################################################################################
# Training Utility classes
########################################################################################################################
class GlobalStep(object):
    """"""
    def __init__(self, gs=-1):
        self._gs = gs

    def step(self):
        self._gs += 1

    def get(self):
        return self._gs

    def set(self, gs):
        self._gs = gs

    def state_dict(self):
        return {"global_step": self._gs}

    def load_state_dict(self, d):
        self.set(d["global_step"])


class DataParallelMock:
    """"""
    def __init__(self, module):
        self.module = module

    def forward(self, *inputs, **kwargs):
        return self.module.forward(*inputs, **kwargs)


########################################################################################################################
# Training Utility Functions
########################################################################################################################
def get_device(device_type, device_ids, verbose=True):
    if device_type == 'cuda' and torch.cuda.is_available():
        if device_ids:
            main_device = min(device_ids)
            name = "cuda:{}".format(main_device)
        else:
            name = "cuda"
    else:
        name = "cpu"
    if verbose:
        print(Fore.YELLOW + 'Using device {}'.format(name) + Fore.RESET)

    return torch.device(name)


def list_files(root, suffix, prefix=True):
    """List all files ending with a suffix at a given root

    Args:
        root (str): Path to directory whose folders need to be listed
        suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
            It uses the Python "str.endswith" method and is passed directly
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the files found
    """
    root = os.path.expanduser(root)
    files = list(
        filter(
            lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix),
            os.listdir(root)
        )
    )

    if prefix is True:
        files = [os.path.join(root, d) for d in files]

    return files


def lin_interpulation(a, b, t):
    return a + (b - a) * t


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def num_parameters(parameter_list):
    n = 0
    for t in parameter_list:
        sz = t.size()
        n_t = 1.
        for s in sz:
            n_t *= s
        n += n_t
    return n


def compute_grad2(out, x_in):
    """"""
    batch_size = x_in[0].size(0)
    grad_out = autograd.grad(
        outputs=out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )
    grad_out2 = [g.pow(2).view(batch_size, -1).sum(1).unsqueeze(1) for g in grad_out]
    reg = torch.cat(grad_out2, dim=1).sum(1)
    return reg


def freeze_module(module):
    for p in module.parameters():
        p.requires_grad = False


def unfreeze_module(module):
    for p in module.parameters():
        p.requires_grad = True


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def save_args(args, outdir):
    json_path = os.path.join(outdir, "args.json")
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    with open(json_path, 'w') as f:
        if isinstance(args, dict):
            json.dump(args, f)
        else:
            json.dump(args.__dict__, f)


def load_args(outdir):
    json_path = os.path.join(outdir, "args.json")
    with open(json_path, 'r') as f:
        args_dict = json.load(f)
    return args_dict


def get_latest_ckpt(ckpt_dir_path, prefix='model_', suffix='.pt'):
    """"""
    model_files = list_files(ckpt_dir_path, (suffix,), prefix=False)
    last_model_file = max(model_files, key=lambda x: int(x[len(prefix):-len(suffix)]))
    return os.path.join(ckpt_dir_path, last_model_file)


def adjust_batch_size(ds_length, curr_batch_size, n_devices):
    batch_size = min(curr_batch_size, ds_length)
    batch_size = int(n_devices * math.floor(batch_size / n_devices))
    assert batch_size % n_devices == 0 and batch_size > 0
    return batch_size


########################################################################################################################
# Unit Test
########################################################################################################################
if __name__ == '__main__':
    def test_compute_grad2():
        batch_size = 4
        x1 = torch.randn((batch_size, 5))
        x2 = torch.randn((batch_size, 5))
        x1.requires_grad_()
        x2.requires_grad_()
        y = x1 + 2 * x2
        y = y * y
        grad2 = compute_grad2(y, (x1, x2))
        print(grad2)

    test_compute_grad2()
