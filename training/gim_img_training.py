# imports
import os
import sys
import itertools
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# local imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(project_root)
from training.utils import get_device, DataParallelMock, adjust_batch_size
from training.logger import Logger
from training.gim_img_trainer import GIMImgTrainer
import models.model_blocks as mb


########################################################################################################################
# Functions
########################################################################################################################
def save_imgs(logger, img_sample, category, k, global_step):
    """"""
    imgs_for_save = ((img_sample[0].clamp(-1, 1) + 1) / 2.0).cpu()
    logger.add_imgs(
        imgs=imgs_for_save,
        category=category,
        k=k,
        global_step=global_step,
    )


def sample_and_save_imgs(device, logger, trainer, ds, ds_prefix, indices, dbg=False):
    """"""
    with torch.no_grad():
        global_step = trainer.module.get_global_step()
        for idx in indices:
            data = ds[idx]
            leaked_sample = data["leaked_sample"].unsqueeze(0).to(device)
            fake_sample = trainer.forward(mode='impersonator_sample', leaked_sample=leaked_sample)
            save_imgs(
                logger=logger,
                img_sample=leaked_sample,
                category="{} imgs_{:04}".format(ds_prefix, idx),
                k="leaked",
                global_step=global_step
            )
            save_imgs(
                logger=logger,
                img_sample=fake_sample,
                category="{} imgs_{:04}".format(ds_prefix, idx),
                k="impersonator",
                global_step=global_step
            )

            if dbg:
                real_sample = data["real_sample"].unsqueeze(0).to(device)
                si_sample = data["si_sample"].unsqueeze(0).to(device)
                save_imgs(
                    logger=logger,
                    img_sample=real_sample,
                    category="{} imgs_{:04}".format(ds_prefix, idx),
                    k="real",
                    global_step=global_step
                )
                save_imgs(
                    logger=logger,
                    img_sample=si_sample,
                    category="{} imgs_{:04}".format(ds_prefix, idx),
                    k="si",
                    global_step=global_step
                )


def im_eval_step(trainer, leaked_sample, si_sample):
    """"""
    trainer.module.impersonator.eval()
    with torch.no_grad():
        loss, fake_sample, au_out = trainer.forward(mode='impersonator_forward', leaked_sample=leaked_sample, si_sample=si_sample)
        loss = loss.mean()
    return loss.detach(), fake_sample.detach(), au_out.detach()


def au_eval_step(trainer, real_sample, fake_sample, si_sample):
    """"""
    trainer.module.authenticator.eval()
    with torch.no_grad():
        loss, loss_on_real, loss_on_fake, reg, out_on_real, out_on_fake, pred_on_real, pred_on_fake, fake_sample = trainer.forward(
            mode='authenticator_forward', fake_sample=fake_sample, real_sample=real_sample, si_sample=si_sample, grad=False
        )
        loss = loss.mean()
    return (loss.detach(), loss_on_real.detach().mean(), loss_on_fake.detach().mean(), reg.detach().mean(),
            out_on_real.detach().mean(), out_on_fake.detach().mean(),
            pred_on_real.detach(), pred_on_fake.detach(), fake_sample.detach())


def eval_step(device, trainer, ds, logger, batch_size):
    """"""
    # stats list
    au_loss_list = []
    au_loss_on_real_list = []
    au_loss_on_fake_list = []
    au_out_on_real_list = []
    au_out_on_fake_list = []
    au_acc_list = []
    au_acc_on_real_list = []
    au_acc_on_fake_list = []
    im_loss_list = []
    global_step = trainer.module.get_global_step()

    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
    num_iters = len(dataloader)
    iter_bar = tqdm(itertools.islice(dataloader, num_iters), total=num_iters, desc='Eval')
    for batch_idx, data_batch in enumerate(iter_bar):
        # data
        real_sample = data_batch["real_sample"].to(device)
        leaked_sample = data_batch["leaked_sample"].to(device)
        si_sample = data_batch["si_sample"].to(device)

        # impersonator train step
        im_loss, fake_sample, _ = im_eval_step(
            trainer=trainer, leaked_sample=leaked_sample, si_sample=si_sample
        )

        # authenticator train step
        au_loss, au_loss_on_real, au_loss_on_fake, au_reg, au_out_on_real, au_out_on_fake, au_pred_on_real, au_pred_on_fake, fake_sample = au_eval_step(
            trainer=trainer, real_sample=real_sample, fake_sample=fake_sample, si_sample=si_sample
        )
        # acc
        au_acc_on_real = au_pred_on_real.to(torch.float).mean()
        au_acc_on_fake = torch.eq(au_pred_on_fake, 0).to(torch.float).mean()
        au_acc = 0.5 * (au_acc_on_real + au_acc_on_fake)

        au_loss_list.append(au_loss.view(1))
        au_loss_on_real_list.append(au_loss_on_real.view(1))
        au_loss_on_fake_list.append(au_loss_on_fake.view(1))
        au_out_on_real_list.append(au_out_on_real.view(1))
        au_out_on_fake_list.append(au_out_on_fake.view(1))
        au_acc_list.append(au_acc.view(1))
        au_acc_on_real_list.append(au_acc_on_real.view(1))
        au_acc_on_fake_list.append(au_acc_on_fake.view(1))
        im_loss_list.append(im_loss.view(1))

    # log
    logger.add_scalar(category='eval losses', k='dis loss', v=torch.cat(au_loss_list).mean().item(), global_step=global_step)
    logger.add_scalar(category='eval losses', k='dis loss on real', v=torch.cat(au_loss_on_real_list).mean().item(), global_step=global_step)
    logger.add_scalar(category='eval losses', k='dis loss on fake', v=torch.cat(au_loss_on_fake_list).mean().item(), global_step=global_step)
    logger.add_scalar(category='eval au out', k='au out on real', v=torch.cat(au_out_on_real_list).mean().item(), global_step=global_step)
    logger.add_scalar(category='eval au out', k='au out on fake', v=torch.cat(au_out_on_fake_list).mean().item(), global_step=global_step)
    logger.add_scalar(category='eval accuracy', k='dis acc', v=torch.cat(au_acc_list).mean().item(), global_step=global_step)
    logger.add_scalar(category='eval accuracy', k='dis acc on real', v=torch.cat(au_acc_on_real_list).mean().item(), global_step=global_step)
    logger.add_scalar(category='eval accuracy', k='dis acc on fake', v=torch.cat(au_acc_on_fake_list).mean().item(), global_step=global_step)
    logger.add_scalar(category='eval losses', k='gen loss', v=torch.cat(im_loss_list).mean().item(), global_step=global_step)


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


def train_epoch(
        device,
        logger,
        epoch,
        trainer,
        train_ds, val_ds,
        train_batch_size, val_batch_size,
        num_workers,
        save_every, eval_every, save_imgs_every, train_eval_indices, val_eval_indices,
        tb_log_every=100, tb_log_enc_every=500,
        n_au_steps=1, dbg=False
):
    """"""
    # log buffers
    au_loss_buffer = []
    au_loss_on_real_buffer = []
    au_loss_on_fake_buffer = []
    au_reg_buffer = []
    au_out_on_real_buffer = []
    au_out_on_fake_buffer = []
    au_pred_on_real_buffer = []
    au_pred_on_fake_buffer = []
    im_loss_buffer = []

    trainloader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    num_iters = 50 if dbg else len(trainloader)
    iter_bar = tqdm(itertools.islice(trainloader, num_iters), total=num_iters, desc='Training')
    for batch_idx, data_batch in enumerate(iter_bar):
        # step
        trainer.module.do_global_step()
        trainer.module.update_learning_rate()

        # data
        real_sample = data_batch["real_sample"].to(device)
        leaked_sample = data_batch["leaked_sample"].to(device)
        si_sample = data_batch["si_sample"].to(device)
        global_step = trainer.module.global_step

        # impersonator train step
        if (global_step + 1) % n_au_steps == 0:
            im_loss, fake_sample, _ = im_train_step(
                trainer=trainer, leaked_sample=leaked_sample, si_sample=si_sample
            )
        else:
            im_loss, fake_sample, _ = im_eval_step(
                trainer=trainer, leaked_sample=leaked_sample, si_sample=si_sample
            )
        im_loss_buffer.append(im_loss.view(1))


        # authenticator train step
        au_loss, au_loss_on_real, au_loss_on_fake, au_reg, au_out_on_real, au_out_on_fake, au_pred_on_real, au_pred_on_fake, fake_sample = au_train_step(
            trainer=trainer, real_sample=real_sample, fake_sample=fake_sample, si_sample=si_sample
        )

        # log stats
        au_loss_buffer.append(au_loss.view(1))
        au_loss_on_real_buffer.append(au_loss_on_real.view(1))
        au_loss_on_fake_buffer.append(au_loss_on_fake.view(1))
        au_reg_buffer.append(au_reg.view(1))
        au_out_on_real_buffer.append(au_out_on_real.view(1))
        au_out_on_fake_buffer.append(au_out_on_fake.view(1))
        au_pred_on_real_buffer.append(au_pred_on_real.view(-1))
        au_pred_on_fake_buffer.append(au_pred_on_fake.view(-1))

        if global_step % tb_log_every == 0:
            # lr
            logger.add_scalar(category='lr', k='au', v=trainer.module.au_lr, global_step=global_step)
            logger.add_scalar(category='lr', k='im', v=trainer.module.im_lr, global_step=global_step)
            logger.add_scalar(category='lr', k='im_lm', v=trainer.module.im_noise_mapping_lr, global_step=global_step)

            # losses
            logger.add_scalar(category='train_losses', k='dis_loss',
                              v=torch.cat(au_loss_buffer).mean().item(), global_step=global_step)
            logger.add_scalar(category='train_losses', k='dis_loss_on_real',
                              v=torch.cat(au_loss_on_real_buffer).mean().item(), global_step=global_step)
            logger.add_scalar(category='train_losses', k='dis_loss_on_fake',
                              v=torch.cat(au_loss_on_fake_buffer).mean().item(), global_step=global_step)
            logger.add_scalar(category='train_losses', k='dis_reg',
                              v=torch.cat(au_reg_buffer).mean().item(), global_step=global_step)

            logger.add_scalar(category='train_au_out', k='au_out_on_real',
                              v=torch.cat(au_out_on_real_buffer).mean().item(), global_step=global_step)
            logger.add_scalar(category='train_au_out', k='au_out_on_fake',
                              v=torch.cat(au_out_on_fake_buffer).mean().item(), global_step=global_step)

            # acc
            au_acc_on_real = torch.cat(au_pred_on_real_buffer).to(torch.float).mean()
            au_acc_on_fake = torch.eq(torch.cat(au_pred_on_fake_buffer), 0).to(torch.float).mean()
            au_acc = 0.5 * (au_acc_on_real + au_acc_on_fake)

            logger.add_scalar(category='train_accuracy', k='dis_acc',
                              v=au_acc.item(), global_step=global_step)
            logger.add_scalar(category='train_accuracy', k='dis_acc_on_real',
                              v=au_acc_on_real.item(), global_step=global_step)
            logger.add_scalar(category='train_accuracy', k='dis_acc_on_fake',
                              v=au_acc_on_fake.item(), global_step=global_step)

            # im
            if len(im_loss_buffer) > 0:
                logger.add_scalar(category='train losses', k='gen loss',
                                  v=torch.cat(im_loss_buffer).mean().item(), global_step=global_step)

            # clear buffers
            au_loss_buffer = []
            au_loss_on_real_buffer = []
            au_loss_on_fake_buffer = []
            au_reg_buffer = []
            au_out_on_real_buffer = []
            au_out_on_fake_buffer = []
            au_pred_on_real_buffer = []
            au_pred_on_fake_buffer = []
            im_loss_buffer = []

        # log encodings
        if global_step % tb_log_enc_every == 0:
            with torch.no_grad():
                # authenticator
                au_real_src = trainer.module.authenticator.src_encode_sample(real_sample)
                au_si_src = trainer.module.authenticator.src_encode_sample(si_sample)
                au_fake_src = trainer.module.authenticator.src_encode_sample(fake_sample)

                au_real_env = trainer.module.authenticator.env_encode_sample(real_sample)
                au_si_env = trainer.module.authenticator.env_encode_sample(si_sample)
                au_fake_env = trainer.module.authenticator.env_encode_sample(fake_sample)

                # mean
                logger.add_scalar(category='train-au_src_mean', k='abs[real-si]',
                                  v=torch.abs(au_real_src.mean(1) - au_si_src.mean(1)).mean().item(),
                                  global_step=global_step)
                logger.add_scalar(category='train-au_src_mean', k='abs[fake-si]',
                                  v=torch.abs(au_fake_src.mean(1) - au_si_src.mean(1)).mean().item(),
                                  global_step=global_step)

                logger.add_scalar(category='train-au_env_mean', k='abs[real-si]',
                                  v=torch.abs(au_real_env.mean(1) - au_si_env.mean(1)).mean().item(),
                                  global_step=global_step)
                logger.add_scalar(category='train-au_env_mean', k='abs[fake-si]',
                                  v=torch.abs(au_fake_env.mean(1) - au_si_env.mean(1)).mean().item(),
                                  global_step=global_step)

                # std
                au_real_src_std = mb.custom_std(au_real_src).mean().item()
                au_si_src_std = mb.custom_std(au_si_src).mean().item()
                au_fake_src_std = mb.custom_std(au_fake_src).mean().item()
                logger.add_scalar(category='train-au_src_std', k='real', v=au_real_src_std, global_step=global_step)
                logger.add_scalar(category='train-au_src_std', k='si', v=au_si_src_std, global_step=global_step)
                logger.add_scalar(category='train-au_src_std', k='fake', v=au_fake_src_std, global_step=global_step)

                au_real_env_std = mb.custom_std(au_real_env).mean().item()
                au_si_env_std = mb.custom_std(au_si_env).mean().item()
                au_fake_env_std = mb.custom_std(au_fake_env).mean().item()
                logger.add_scalar(category='train-au_env_std', k='real', v=au_real_env_std, global_step=global_step)
                logger.add_scalar(category='train-au_env_std', k='si', v=au_si_env_std, global_step=global_step)
                logger.add_scalar(category='train-au_env_std', k='fake', v=au_fake_env_std, global_step=global_step)

        if (global_step % save_every == 0):
            trainer.module.save(epoch=epoch)

        if global_step % save_imgs_every == 0:
            sample_and_save_imgs(
                device=device, logger=logger, trainer=trainer, ds=train_ds, ds_prefix='train', indices=train_eval_indices, dbg=dbg
            )
            sample_and_save_imgs(
                device=device, logger=logger, trainer=trainer, ds=val_ds, ds_prefix='val', indices=val_eval_indices, dbg=dbg
            )

        if global_step % eval_every == 0:
            eval_step(device=device, trainer=trainer, ds=val_ds, logger=logger, batch_size=val_batch_size)


def train_gim_imgs(
        device_name,
        device_ids,
        outdir,
        train_ds, val_ds,
        authenticator,
        impersonator,
        m, n, k,
        reg_param, remove_noise_mean,
        au_lr, im_lr, beta1, beta2, env_noise_mapping_lr,
        lr_gamma, milestones,
        resume_from_ckpt, n_epochs, batch_size, num_workers,
        save_every, eval_every, save_imgs_every,
        train_eval_indices, val_eval_indices,
        n_au_steps=1, dbg=False
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
    trainer = GIMImgTrainer(
        outdir=outdir,
        m=m, n=n, k=k,
        authenticator=authenticator, impersonator=impersonator,
        au_lr=au_lr, im_lr=im_lr, env_noise_mapping_lr=env_noise_mapping_lr,
        beta1=beta1, beta2=beta2,
        lr_milestones=milestones, lr_gamma=lr_gamma,
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
    epoch_bar = tqdm(range(n_epochs), "Epochs")
    for ep in epoch_bar:
        try:
            train_epoch(
                device=device,
                logger=logger,
                epoch=ep,
                trainer=trainer,
                train_ds=train_ds, val_ds=val_ds,
                train_batch_size=adjust_batch_size(len(train_ds), batch_size, n_devices),
                val_batch_size=adjust_batch_size(len(val_ds), batch_size, n_devices),
                num_workers=num_workers,
                save_every=save_every, eval_every=eval_every,
                save_imgs_every=save_imgs_every,
                train_eval_indices=train_eval_indices, val_eval_indices=val_eval_indices,
                n_au_steps=n_au_steps, dbg=dbg
            )

        except KeyboardInterrupt:
            print()
            print('KeyboardInterrupt')
            print("Saving checkpoint...")
            print()
            trainer.module.save(ep)
            break

        except PermissionError as pe:
            print()
            print('PermissionError')
            print(pe)
            print("Saving checkpoint...")
            print()
            trainer.module.save(ep)
            continue