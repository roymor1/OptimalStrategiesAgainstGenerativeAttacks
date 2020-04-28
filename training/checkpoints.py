# imports
import os
import torch


########################################################################################################################
# Classes
########################################################################################################################
class CheckpointIO:
    """"""
    def __init__(self, checkpoint_dir, **kwargs):
        self.module_dict = kwargs
        self.checkpoint_dir = checkpoint_dir

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    def register_modules(self, **kwargs):
        self.module_dict.update(kwargs)

    def save(self, global_step, last_epoch, filename):
        filename = os.path.join(self.checkpoint_dir, filename)

        outdict = {'global_step': global_step, "last_epoch": last_epoch}
        for k, v in self.module_dict.items():
            outdict[k] = v.state_dict()
        torch.save(outdict, filename)

    def load(self, filepath):
        if os.path.exists(filepath):
            print('=> Loading checkpoint...')
            out_dict = torch.load(filepath, map_location='cpu')
            global_step = out_dict['global_step']
            last_epoch = out_dict['last_epoch']
            for k, v in self.module_dict.items():
                if k in out_dict:
                    v.load_state_dict(out_dict[k])
                else:
                    print('Warning: Could not find %s in checkpoint!' % k)
        else:
            global_step = -1
            last_epoch = -1

        return global_step, last_epoch

