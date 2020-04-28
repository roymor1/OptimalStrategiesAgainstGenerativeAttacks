import pickle
import os
import torch
import torchvision
from torchvision import transforms
import tensorboardX


########################################################################################################################
# Classes
########################################################################################################################
class Logger(object):
    """"""
    def __init__(self, log_dir='./logs', img_dir='./imgs', tensorboard_dir=None):
        self.stats = dict()
        self.log_dir = log_dir
        self.img_dir = img_dir

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        self.setup_monitoring(tensorboard_dir)

    def setup_monitoring(self, tensorboard_dir):
        self.monitoring_dir = tensorboard_dir
        self.tb = tensorboardX.SummaryWriter(tensorboard_dir)

    def add_scalar(self, category, k, v, global_step):
        if category not in self.stats:
            self.stats[category] = {}

        if k not in self.stats[category]:
            self.stats[category][k] = []

        self.stats[category][k].append((global_step, v))

        k_name = '%s/%s' % (category, k)
        self.tb.add_scalar(k_name, v, global_step)

    def add_imgs(self, imgs, category, k, global_step, nrow=5):
        """"""
        outdir = os.path.join(self.img_dir, category, k)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        outfile = os.path.join(outdir, '%08d.png' % global_step)
        grid_img = torchvision.utils.make_grid(imgs, nrow=nrow)
        torchvision.utils.save_image(grid_img.clone(), outfile, nrow=nrow)
        tag_full = '%s/%s' % (category, k)
        self.tb.add_image(tag=tag_full, img_tensor=grid_img, global_step=global_step)

    def add_figure(self, fig, category, k, global_step):
        """"""
        outdir = os.path.join(self.img_dir, category, k)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        figure_path = os.path.join(outdir, '%08d.png' % global_step)
        fig.savefig(figure_path)

        tag_full = '%s/%s' % (category, k)
        self.tb.add_figure(tag=tag_full, figure=fig, global_step=global_step)

    def add_embeddings(self, embs, label_imgs, tag, global_step):
        """"""
        self.tb.add_embedding(tag=tag, mat=embs, label_img=label_imgs, global_step=global_step)

    def get_last_scalar(self, category, k, default=0.):
        if category not in self.stats:
            return default
        elif k not in self.stats[category]:
            return default
        else:
            return self.stats[category][k][-1][1]

    def save_stats(self, filename):
        filename = os.path.join(self.log_dir, filename)
        with open(filename, 'wb') as f:
            pickle.dump(self.stats, f)

    def load_stats(self, filename):
        filename = os.path.join(self.log_dir, filename)
        if not os.path.exists(filename):
            print('Warning: file "%s" does not exist!' % filename)
            return

        try:
            with open(filename, 'rb') as f:
                self.stats = pickle.load(f)
        except EOFError:
            print('Warning: log file corrupted!')
