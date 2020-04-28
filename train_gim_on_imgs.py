# imports
import os
import sys
import argparse
import torch
torch.manual_seed(1)


# local imports
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_root)
from models.gim_img_models import get_au, get_im
from training.gim_img_training import train_gim_imgs
from training.utils import save_args
from data_handling.img_datasets import ImgGIMDataSet, OmniglotGIMDataSet


# app
def main(args):
    """"""
    # models
    au = get_au(
        img_size=args.img_size,
        img_channels=args.img_channels,
        style_dim=args.style_dim
    )
    im = get_im(
        img_size=args.img_size,
        img_channels=args.img_channels,
        style_dim=args.style_dim,
        use_img_att=args.use_img_att,
        num_env_noise_layers=args.num_env_noise_layers
    )

    # load pretrained
    if args.pretrained:
        models_state_dict = torch.load(args.pretrained, map_location='cpu')
        au.load_state_dict(models_state_dict['authenticator'])
        im.load_state_dict(models_state_dict['impersonator'])

    # data
    if args.dataset_type == 'omniglot':
        train_ds = OmniglotGIMDataSet(
            root=args.dataset_root,
            split='train',
            img_channels=args.img_channels,
            img_size=args.img_size,
            m=args.m, n=args.n, si=args.k, example_cnt_per_class=args.ds_n_examples_per_cls,
        )
        val_ds = OmniglotGIMDataSet(
            root=args.dataset_root,
            split='val',
            img_channels=args.img_channels,
            img_size=args.img_size,
            m=args.m, n=args.n, si=args.k, example_cnt_per_class=1,
        )
    elif args.dataset_type == 'voxceleb2':
        train_ds = ImgGIMDataSet(
            root=args.dataset_root,
            split='train',
            img_channels=args.img_channels,
            img_size=args.img_size,
            m=args.m, n=args.n, si=args.k, example_cnt_per_class=args.ds_n_examples_per_cls,
            hierarchical=True, mirror=True
        )
        val_ds = ImgGIMDataSet(
            root=args.dataset_root,
            split='val',
            img_channels=args.img_channels,
            img_size=args.img_size,
            m=args.m, n=args.n, si=args.k, example_cnt_per_class=1,
            hierarchical=True, mirror=True
        )
    else:
        raise ValueError("Supports only dataset_type in ['omniglot','voxceleb2']")

    # train
    train_gim_imgs(
        device_name=args.device,
        device_ids=args.device_ids,
        outdir=args.outdir,
        train_ds=train_ds, val_ds=val_ds,
        authenticator=au,
        impersonator=im,
        m=args.m, n=args.n, k=args.k,
        reg_param=args.reg_param,
        remove_noise_mean=args.remove_noise_mean,
        au_lr=args.au_lr,
        im_lr=args.im_lr,
        beta1=args.beta1, beta2=args.beta2,
        env_noise_mapping_lr=args.env_noise_mapping_lr,
        lr_gamma=args.lr_gamma,
        milestones=args.milestones,
        resume_from_ckpt=args.resume_from_ckpt,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        save_every=args.save_every,
        eval_every=args.eval_every,
        save_imgs_every=args.save_imgs_every,
        train_eval_indices=list(range(0, len(train_ds), int(len(train_ds) / 10))),
        val_eval_indices=list(range(0, len(val_ds), int(len(val_ds) / 10))),
        n_au_steps=args.n_au_steps, dbg=args.dbg
    )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', default='cuda',
                        help="cuda or cpu")
    parser.add_argument('--device_ids', nargs='+', type=int, default=[0],
                        help='A list of device ids for the GPUS to be used. '
                             'E.g., if using GPUs 1,3,5,8, use: --device_ids 1 3 5 8. '
                             'Only relevant when using cuda.')
    parser.add_argument('-o', '--outdir', default='./gim_imgs_outdir/',
                        help='Output directory for the experiment ')
    parser.add_argument('--dataset_root', required=True,
                        help='Path to dataset root dir')
    parser.add_argument('--dataset_type', default='omniglot',
                        help='Options are omniglot or voxceleb2')
    parser.add_argument('--ckpt_dir_name', default='ckpts')
    parser.add_argument('-r', '--resume_from_ckpt', default=None,
                        help='Path to a checkpoint from which to resume training')
    parser.add_argument('--pretrained', default=None,
                        help='Path to pretrained checkpoint to use for model initialization')
    parser.add_argument('--n_epochs', type=int, default=100000,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int ,default=128)
    parser.add_argument('--num_workers', type=int ,default=4)
    parser.add_argument('--ds_n_examples_per_cls', type=int ,default=100,
                        help='Number of examples per class in an epoch')
    parser.add_argument('--m', type=int, default=1, help='m: The number of leaked images')
    parser.add_argument('--n', type=int, default=5, help='n: The number of test images')
    parser.add_argument('--k', type=int, default=5, help='k: The number of registration images')
    parser.add_argument('--img_channels', type=int, default=1,
                        help='Number of image channels. 1 for omniglot, 3 for voxceleb2')
    parser.add_argument('--img_size', type=int, default=32,
                        help='Image size. 32 for omniglot, 64 for voxceleb2')
    parser.add_argument('--style_dim', type=int, default=512)
    parser.add_argument('--num_env_noise_layers', type=int, default=4)
    parser.add_argument('--au_lr', type=float, default=1e-6,
                        help='Learning rate for the authenticator. Use 1e-6 for omniglot and 1e-4 for voxceleb2')
    parser.add_argument('--im_lr', type=float, default=1e-5,
                        help='Learning rate for the attacker (or impersonator). Use 1e-5 for omniglot, 1e-4 for voxceleb2')
    parser.add_argument('--beta1', type=float, default=0.,
                        help="beta1 for the Adam optimizer")
    parser.add_argument('--beta2', type=float, default=0.99,
                        help="beta2 for the Adam optimizer")
    parser.add_argument('--env_noise_mapping_lr', type=float, default=1e-7,
                        help='Learning rate for the noise mapping module. Use 1e-7 for omniglot, 1e-6 for voxceleb2')
    parser.add_argument('--lr_gamma', type=float, default=0.3)
    parser.add_argument('--milestones', type=int, nargs='+', default=[])
    parser.add_argument('--reg_param', type=float, default=0.,
                        help='GAN regularization coefficient. Use 0. for omniglot, 10. for voxceleb2')
    parser.add_argument('--remove_noise_mean', type=lambda x: bool(int(x)), default=True)
    parser.add_argument('--use_img_att', type=lambda x: bool(int(x)), default=False)
    parser.add_argument('--save_every', type=int, default=10000)
    parser.add_argument('--eval_every', type=int ,default=500)
    parser.add_argument('--save_imgs_every', type=int ,default=500)
    parser.add_argument('--n_au_steps', type=int ,default=1)
    parser.add_argument('-dbg', action='store_true')
    return parser.parse_args()


# app
if __name__ == '__main__':
    args = get_args()
    save_args(args=args, outdir=args.outdir)
    main(args)
