# imports
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# local imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(project_root)
from training.utils import GlobalStep, compute_grad2, num_parameters
from training.checkpoints import CheckpointIO


########################################################################################################################
# Trainer
########################################################################################################################
class GIMGaussianTrainer(nn.Module):
    """"""
    CHECKPOINT_DIR = "ckpts"

    def __init__(
            self,
            outdir,
            m, n, k,
            authenticator, impersonator,
            au_lr, im_lr,
            reg_param=0., remove_noise_mean=True
    ):
        """"""
        super().__init__()
        # game settings
        self.m = m
        self.n = n
        self.k = k

        # agents
        self.authenticator = authenticator
        self.impersonator = impersonator

        # training_legacy & optimization
        self._global_step = GlobalStep()
        self.reg_param = reg_param
        self.remove_noise_mean = remove_noise_mean

        self.authenticator_opt = optim.Adam(self.authenticator.parameters(), lr=au_lr)
        self.impersonator_opt = optim.Adam(self.impersonator.parameters(), lr=im_lr)

        print("Authenticator has {} parameters".format(num_parameters(self.authenticator.parameters())))
        print("impersonator has {} parameters".format(num_parameters(self.impersonator.parameters())))

        # checkpoint
        self.checkpoint_dir = os.path.join(outdir, self.CHECKPOINT_DIR)
        self.checkpoint_io = CheckpointIO(checkpoint_dir=self.checkpoint_dir)

        # Register modules to checkpoint
        self.checkpoint_io.register_modules(
            authenticator=self.authenticator,
            impersonator=self.impersonator,
            authenticator_opt=self.authenticator_opt,
            impersonator_opt=self.impersonator_opt,
            global_step=self._global_step
        )

    def forward(self, mode, **kwargs):
        """"""
        if mode == "authenticator_forward":
            return self.authenticator_forward(**kwargs)
        elif mode == "impersonator_forward":
            return self.impersonator_forward(**kwargs)
        elif mode == "impersonator_sample":
            return self.impersonator_sample(**kwargs)
        else:
            raise ValueError("unsupported mode")

    def gan_loss(self, dis_out, target, reduce=False):
        """"""
        targets = dis_out.new_full(size=dis_out.size(), fill_value=target).detach()
        loss = F.binary_cross_entropy_with_logits(dis_out, targets, reduce=reduce)
        return loss.squeeze()

    def authenticator_forward(self, fake_sample, real_sample, si_sample, grad=True):
        """"""
        # reg setup
        if self.reg_param > 0:
            real_sample.requires_grad_()
            si_sample.requires_grad_()

        # on real sample
        out_on_real = self.authenticator(test_sample=real_sample, si_sample=si_sample)
        loss_on_real = self.gan_loss(dis_out=out_on_real, target=1.)
        if grad and self.reg_param > 0:
            reg = self.reg_param * compute_grad2(out_on_real, (real_sample, si_sample))
        else:
            reg = torch.zeros_like(loss_on_real)

        # on fake sample
        out_on_fake = self.authenticator(test_sample=fake_sample, si_sample=si_sample)
        loss_on_fake = self.gan_loss(dis_out=out_on_fake, target=0.)

        # predictions:
        with torch.no_grad():
            pred_on_real = torch.ge(out_on_real.detach(), 0)
            pred_on_fake = torch.ge(out_on_fake.detach(), 0)

        # loss
        loss = loss_on_real + loss_on_fake + reg
        return loss, loss_on_real.detach(), loss_on_fake.detach(), reg, out_on_real.detach(), out_on_fake.detach(), pred_on_real.detach(), pred_on_fake.detach(), fake_sample.detach()

    def impersonator_forward(self, leaked_sample, si_sample):
        """"""
        fake_sample = self.impersonator(leaked_sample=leaked_sample, n=self.n, remove_noise_mean=self.remove_noise_mean)
        auth_out = self.authenticator(test_sample=fake_sample, si_sample=si_sample)
        loss = self.gan_loss(dis_out=auth_out, target=1.)
        return loss, fake_sample, auth_out

    def impersonator_sample(self, leaked_sample):
        """"""
        with torch.no_grad():
            fake_sample = self.impersonator(leaked_sample=leaked_sample, n=self.n, remove_noise_mean=self.remove_noise_mean)
        return fake_sample

    # save & restore
    def resume_from_ckpt(self, ckpt_path):
        """"""
        _, _ = self.checkpoint_io.load(ckpt_path)
        print('Resuming training_legacy from iteration {}'.format(self.get_global_step()))

    def save(self):
        """"""
        print()
        print("Saving checkpoint...")
        print()
        self.checkpoint_io.save(
            global_step=self.get_global_step(),
            last_epoch=1,
            filename="model_{:08}.pt".format(self.get_global_step())
        )

    # getters and setters
    def get_global_step(self):
        return self._global_step.get()

    def do_global_step(self):
        return self._global_step.step()

    @property
    def global_step(self):
        return self.get_global_step()