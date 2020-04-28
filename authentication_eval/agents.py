# imports
import os
import sys
import random
import torch


# local imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(project_root)


########################################################################################################################
# Authenticator
########################################################################################################################
class Authenticator:
    """"""
    def __init__(self, au_model_func ,th=0.):
        self.au_model_func = au_model_func
        self.th = th

    def act(self, test_sample, si_sample):
        """"""
        out = self.au_model_func(test_sample=test_sample, si_sample=si_sample)
        pred = torch.ge(out, self.th).to(torch.long)
        return out, pred


########################################################################################################################
# Impersonator
########################################################################################################################
class Impersonator:
    """"""
    def __init__(self, im_model_func):
        self.im_model_func = im_model_func

    def act(self, leaked_sample, n):
        """"""
        fake_sample = self.im_model_func(leaked_sample=leaked_sample, n=n)
        return fake_sample


########################################################################################################################
# Impersonator Functions
########################################################################################################################
def replay_impersonator(leaked_sample, n):
    """"""
    m = leaked_sample.size(1)
    fake_sample = torch.cat([leaked_sample[:, random.randrange(m)].unsqueeze(1) for _ in range(n)], dim=1)
    return fake_sample.to(leaked_sample.device)


def rand_source_impersonator(leaked_sample, n, gim_ds):
    """"""
    batch_size = leaked_sample.size(0)
    fake_sample = []
    for _ in range(batch_size):
        idx = random.randint(0, len(gim_ds) - 1)
        fake_sample.append(gim_ds[idx]["real_sample"])
    fake_sample = torch.stack(fake_sample, dim=0)
    assert fake_sample.size(1) == n
    return fake_sample.to(leaked_sample.device)


########################################################################################################################
# Unit test
########################################################################################################################