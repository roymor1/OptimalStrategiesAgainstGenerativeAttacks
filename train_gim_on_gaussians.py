# imports
import os
import sys
import argparse
import torch
torch.manual_seed(1)


# local imports
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_root)
from training.utils import save_args
from training.gim_gaussian_training import train_gim_gaussian
from models.gim_gaussian_models import get_au, get_im


# app
def main(args):
    """"""
    # models
    au = get_au(src_dim=args.src_dim)
    im = get_im(src_dim=args.src_dim)

    # load pretrained
    if args.pretrained:
        models_state_dict = torch.load(args.pretrained, map_location='cpu')
        au.load_state_dict(models_state_dict['authenticator'])
        im.load_state_dict(models_state_dict['impersonator'])

    # train
    train_gim_gaussian(
        device_name=args.device,
        device_ids=args.device_ids,
        outdir=args.outdir,
        authenticator=au,
        impersonator=im,
        m=args.m, n=args.n, k=args.k,
        src_dim=args.src_dim,
        src_sigma=args.src_sigma,
        prior_sigma=args.prior_sigma,
        reg_param=args.reg_param,
        remove_noise_mean=args.remove_noise_mean,
        au_lr=args.au_lr,
        im_lr=args.im_lr,
        resume_from_ckpt=args.resume_from_ckpt,
        n_iters=args.n_iters,
        batch_size=args.batch_size,
        save_every=args.save_every,
        save_stats_every=args.save_stats_every
    )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', default='cuda',
                        help='cuda or cpu')
    parser.add_argument('--device_ids', nargs='+', type=int, default=[0],
                        help='A list of device ids for the GPUS to be used. '
                             'E.g., if using GPUs 1,3,5,8, use: --device_ids 1 3 5 8. '
                             'Only relevant when using cuda.')
    parser.add_argument('-o', '--outdir', default='./gim_gaussians_outdir/',
                        help='Output directory for the experiment ')
    parser.add_argument('--ckpt_dir_name', default='ckpts')
    parser.add_argument('-r', '--resume_from_ckpt', default=None,
                        help='Path to a checkpoint from which to resume training')
    parser.add_argument('--pretrained',
                        default=None,
                        help='Path to pretrained checkpoint to use for model initialization')
    parser.add_argument('--n_iters', type=int, default=500000, help='Number of training iterations.')
    parser.add_argument('--batch_size', type=int ,default=4096)
    parser.add_argument('--m', type=int, default=1, help='m: The number of leaked images')
    parser.add_argument('--n', type=int, default=10, help='n: The number of test images')
    parser.add_argument('--k', type=int, default=10, help='k: The number of registration images')

    parser.add_argument('--prior_sigma', type=float, default=10.,
                        help='The standard deviation of Q, the prior distribution over sources.')
    parser.add_argument('--src_sigma', type=float, default=1.,
                        help="The known standard deviation of the sources' diagonal covariance.")
    parser.add_argument('--src_dim', type=int, default=1,
                        help="The dimension of source observations")
    parser.add_argument('--au_lr', type=float, default=0.0001, help='Learning rate for the authenticator')
    parser.add_argument('--im_lr', type=float, default=0.0001, help='Learning rate for the attacker (impersonator)')
    parser.add_argument('--reg_param', type=float, default=0., help='GAN regularization coefficient. Must be set to 0')
    parser.add_argument('--remove_noise_mean', type=lambda x: bool(int(x)), default=True)

    parser.add_argument('--save_every', type=int, default=100000)
    parser.add_argument('--eval_every', type=int ,default=1000)
    parser.add_argument('--save_stats_every', type=int ,default=100)
    return parser.parse_args()


# app
if __name__ == '__main__':
    args = get_args()
    save_args(args=args, outdir=args.outdir)
    main(args)
