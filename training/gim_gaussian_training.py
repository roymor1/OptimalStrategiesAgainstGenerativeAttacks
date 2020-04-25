# imports
import os
import sys
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F


# local imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(project_root)
from training.utils import get_device, DataParallelMock
from training.logger import Logger
from training.gim_gaussian_trainer import GIMGaussianTrainer
import models.model_blocks as mb
# from models.gim_basic_models import


# functions
def im_train_step(trainer, leaked_sample, si_sample):
    """"""
    trainer.module.impersonator.train()
    trainer.module.impersonator_opt.zero_grad()

    loss, fake_sample, au_out = trainer.forward(mode='impersonator_forward', leaked_sample=leaked_sample, si_sample=si_sample)
    loss = loss.mean()
    loss.backward()
    trainer.module.impersonator_opt.step()
    return loss.detach(), fake_sample.detach(), au_out.detach()


def au_train_step(trainer, real_sample, fake_sample, si_sample):
    """"""
    trainer.module.authenticator.train()
    trainer.module.authenticator_opt.zero_grad()

    loss, loss_on_real, loss_on_fake, reg, out_on_real, out_on_fake, pred_on_real, pred_on_fake, fake_sample = trainer.forward(
        mode='authenticator_forward', fake_sample=fake_sample, real_sample=real_sample, si_sample=si_sample
    )
    loss = loss.mean()
    loss.backward()
    trainer.module.authenticator_opt.step()

    return (loss.detach(), loss_on_real.detach().mean(), loss_on_fake.detach().mean(), reg.detach().mean(),
            out_on_real.detach().mean(), out_on_fake.detach().mean(),
            pred_on_real.detach(), pred_on_fake.detach(), fake_sample.detach())


def train(
        device,
        trainer,
        logger,
        n_iters,
        batch_size,
        src_dim,
        src_sigma,
        prior_sigma,
        save_stats_every,
        save_every
):
    """"""
    iter_bar = tqdm(range(n_iters), total=n_iters, desc='Training')
    for _ in iter_bar:
        # step
        trainer.module.do_global_step()
        m = trainer.module.m
        n = trainer.module.n
        k = trainer.module.k

        # data
        mu = torch.normal(mean=torch.zeros((batch_size, src_dim)),
                          std=torch.full((batch_size, src_dim), prior_sigma)).to(device)
        sigma = torch.full((batch_size, src_dim), src_sigma).to(device)
        real_sample = torch.normal(
            mean=mu.unsqueeze(1).repeat(1, n, 1),
            std=sigma.unsqueeze(1).repeat(1, n, 1)
        ).to(device)
        leaked_sample = torch.normal(
            mean=mu.unsqueeze(1).repeat(1, m, 1),
            std=sigma.unsqueeze(1).repeat(1, m, 1)
        ).to(device)
        si_sample = torch.normal(
            mean=mu.unsqueeze(1).repeat(1, k, 1),
            std=sigma.unsqueeze(1).repeat(1, k, 1)
        ).to(device)
        global_step = trainer.module.get_global_step()


        # impersonator train step
        im_loss, fake_sample, _ = im_train_step(trainer=trainer, leaked_sample=leaked_sample, si_sample=si_sample)
        logger.add_scalar(category='train losses', k='im loss', v=im_loss.item(), global_step=global_step)

        # authenticator train step
        au_loss, au_loss_on_real, au_loss_on_fake, au_reg, au_out_on_real, au_out_on_fake, au_pred_on_real, au_pred_on_fake, fake_sample = au_train_step(
            trainer=trainer, real_sample=real_sample, fake_sample=fake_sample, si_sample=si_sample
        )

        # log stats
        au_acc_on_real = au_pred_on_real.to(torch.float).mean()
        au_acc_on_fake = torch.eq(au_pred_on_fake, 0).to(torch.float).mean()
        au_acc = 0.5 * (au_acc_on_real + au_acc_on_fake)

        logger.add_scalar(category='train losses', k='au loss', v=au_loss.item(), global_step=global_step)
        logger.add_scalar(category='train losses', k='au loss on real', v=au_loss_on_real.item(), global_step=global_step)
        logger.add_scalar(category='train losses', k='au loss on fake', v=au_loss_on_fake.item(), global_step=global_step)
        logger.add_scalar(category='train losses', k='au reg', v=au_reg.item(), global_step=global_step)
        logger.add_scalar(category='train au out', k='au out on real', v=au_out_on_real.item(), global_step=global_step)
        logger.add_scalar(category='train au out', k='au out on fake', v=au_out_on_fake.item(), global_step=global_step)

        logger.add_scalar(category='train accuracy', k='au acc', v=au_acc.item(), global_step=global_step)
        logger.add_scalar(category='train accuracy', k='au acc on real', v=au_acc_on_real.item(), global_step=global_step)
        logger.add_scalar(category='train accuracy', k='au acc on fake', v=au_acc_on_fake.item(), global_step=global_step)


        # log stats
        if global_step % save_stats_every == 0:
            with torch.no_grad():
                logger.add_scalar(
                    category='im distances',
                    k='l1_dist_from_leaked_sample_mean',
                    v=F.l1_loss(fake_sample.mean(dim=1), leaked_sample.mean(dim=1)).item(),
                    global_step=global_step
                )
                logger.add_scalar(
                    category='im distances',
                    k='l1_dist_from_gt_sample_mean',
                    v=F.l1_loss(fake_sample.mean(dim=1), mu).item(),
                    global_step=global_step
                )
                logger.add_scalar(
                    category='im distances',
                    k='l1_dist_from_gt_std',
                    v=F.l1_loss(mb.custom_std(fake_sample), sigma).item(),
                    global_step=global_step
                )
                logger.add_scalar(
                    category='real distances',
                    k='l1_dist_from_gt_sample_mean',
                    v=F.l1_loss(real_sample.mean(dim=1), mu).item(),
                    global_step=global_step
                )
                logger.add_scalar(
                    category='real distances',
                    k='l1_dist_from_gt_std',
                    v=F.l1_loss(mb.custom_std(real_sample), sigma).item(),
                    global_step=global_step
                )
        # save
        if (global_step % save_every == 0):
            trainer.module.save()


def train_gim_gaussian(
        device_name,
        device_ids,
        outdir,
        authenticator,
        impersonator,
        m, n, k,
        src_dim, src_sigma, prior_sigma,
        reg_param, remove_noise_mean,
        au_lr, im_lr,
        resume_from_ckpt, n_iters, batch_size,
        save_every, save_stats_every,
):
    """"""
    # device
    device = get_device(device_type=device_name, device_ids=device_ids)
    n_devices = len(device_ids)
    assert batch_size % n_devices == 0

    # logger
    logger = Logger(
        log_dir=os.path.join(outdir, 'logs'),
        img_dir=os.path.join(outdir, 'imgs'),
        tensorboard_dir=os.path.join(outdir, 'tb')
    )

    # models
    authenticator = authenticator.to(device)
    impersonator = impersonator.to(device)

    # trainer
    trainer = GIMGaussianTrainer(
        outdir=outdir,
        m=m, n=n, k=k,
        authenticator=authenticator, impersonator=impersonator,
        au_lr=au_lr, im_lr=im_lr,
        reg_param=reg_param, remove_noise_mean=remove_noise_mean
    ).to(device)


    if resume_from_ckpt:
        trainer.resume_from_ckpt(ckpt_path=resume_from_ckpt)
        trainer.to(device)

    # set parallel training
    if device_name == 'cuda':
        print("using devices: {}".format(device_ids))
        trainer = nn.DataParallel(trainer, device_ids=device_ids)
    else:
        trainer = DataParallelMock(trainer)

    # training
    try:
        train(
            device=device,
            trainer=trainer,
            logger=logger,
            n_iters=n_iters,
            batch_size=batch_size,
            src_dim=src_dim,
            src_sigma=src_sigma,
            prior_sigma=prior_sigma,
            save_stats_every=save_stats_every,
            save_every=save_every
        )
    except KeyboardInterrupt:
        print()
        print('KeyboardInterrupt')
        print("Saving checkpoint...")
        print()
        trainer.module.save()

    except PermissionError as pe:
        print()
        print('PermissionError')
        print(pe)
        print("Saving checkpoint...")
        print()
        trainer.module.save()