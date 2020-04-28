# imports
import os
import sys
import argparse
import pandas as pd
import torch
torch.manual_seed(1)


# local imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(project_root)
from models.gim_img_models import get_im, get_au
from data_handling.img_datasets import OmniglotGIMDataSet, ImgGIMDataSet
from training.utils import load_args, get_device, get_latest_ckpt
from authentication_eval.agents import Authenticator, Impersonator, replay_impersonator, rand_source_impersonator
from authentication_eval.authentication_score import eval_authenticator_and_impersonator
from baselines.siamese.models import ProtonetEmbeddingNet, SiameseNet
from baselines.arcface.models import Backbone, ArcFace


########################################################################################################################
# functions
########################################################################################################################
def get_au_function(au):
    """"""
    def au_model_func(test_sample, si_sample):
        with torch.no_grad():
            # encode
            au_si_src = au.src_encode_sample(si_sample)
            au_si_env = au.env_encode_sample(si_sample)

            au_test_src = au.src_encode_sample(test_sample)
            au_test_env = au.env_encode_sample(test_sample)

            # on real sample
            out = au.dis(
                test_src=au_test_src,
                test_env=au_test_env,
                si_src=au_si_src,
                si_env=au_si_env
            )
        return out.detach()
    return au_model_func


def get_siamese_au_function(model):
    def au_model_func(test_sample, si_sample):
        model.train(mode=False)
        with torch.no_grad():
            si_size = si_sample.size()
            test_size = test_sample.size()
            si_emb = model.encode(
                si_sample.view(si_size[0] * si_size[1], *si_size[2:])
            ).view(si_size[0], si_size[1], -1).mean(dim=1)
            test_emb = model.encode(
                test_sample.contiguous().view(test_size[0] * test_size[1], *test_size[2:])
            ).contiguous().view(test_size[0], test_size[1], -1).mean(dim=1)
            logits = model.classify(si_emb, test_emb)
        return logits.squeeze().detach()
    return au_model_func


def get_arcface_au_function(arcface):
    def au_model_func(test_sample, si_sample):
        arcface.train(mode=False)
        with torch.no_grad():
            x1 = test_sample.mean(1)
            x2 = si_sample.mean(1)
            score, _ = arcface.predict(x1=x1, x2=x2)
        return score.detach()
    return au_model_func


def get_im_function(im, args_dict):
    def im_model_func(leaked_sample, n):
        with torch.no_grad():
            fake_sample = im.forward(leaked_sample=leaked_sample, n=n, remove_noise_mean=args_dict['remove_noise_mean'])
        return fake_sample.detach()
    return im_model_func


def get_gim_authenticator(device, ckpt_path, args_dict):
    """"""
    au = get_au(
        img_size=args_dict['img_size'],
        img_channels=args_dict['img_channels'],
        style_dim=args_dict['style_dim']
    )
    au.load_state_dict(torch.load(ckpt_path, map_location='cpu')['authenticator'])
    au = au.to(device)
    return Authenticator(get_au_function(au))


def get_gim_impersonator(device, ckpt_path, args_dict):
    """"""
    im = get_im(
        img_size=args_dict['img_size'],
        img_channels=args_dict['img_channels'],
        style_dim=args_dict['style_dim'],
        use_img_att=args_dict['use_img_att'],
        num_env_noise_layers=args_dict['num_env_noise_layers']
    )
    im.load_state_dict(torch.load(ckpt_path, map_location='cpu')['impersonator'])
    im = im.to(device)
    return Impersonator(get_im_function(im, args_dict))


def get_siamese_authenticator(device, ckpt_path, args_dict):
    """"""
    encoder = ProtonetEmbeddingNet(1, 32)
    model = SiameseNet(encoder, encoder.embedding_dim)
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu')['model'])
    model = model.to(device)
    return Authenticator(get_siamese_au_function(model))


def get_arcface_authenticator(device, ckpt_path, args_dict):
    """"""
    state_dict = torch.load(ckpt_path, map_location='cpu')['arcface']
    n_classes = state_dict['head.kernel'].size(-1)
    emb_model = Backbone(num_layers=args_dict['num_layers'], drop_ratio=args_dict['dropout'], mode='ir_se',
                         img_size=args_dict['img_size'], img_channels=args_dict['img_channels'])
    arcface = ArcFace(emb_model=emb_model, embedding_size=args_dict['emb_dim'], n_classes=n_classes)
    arcface.load_state_dict(state_dict)
    arcface.th = args_dict['th']
    arcface = arcface.to(device)
    return Authenticator(get_arcface_au_function(arcface), th=arcface.th)


def get_authenticator(device, au_type, ckpt_path, args_dict):
    """"""
    if au_type == 'gim':
        return get_gim_authenticator(device=device, ckpt_path=ckpt_path, args_dict=args_dict)
    elif au_type == 'siamese':
        return get_siamese_authenticator(device=device, ckpt_path=ckpt_path, args_dict=args_dict)
    elif au_type == 'arcface':
        return get_arcface_authenticator(device=device, ckpt_path=ckpt_path, args_dict=args_dict)
    else:
        assert False, "unsupported authenticator type"


def get_impersonator(device, im_type, ckpt_path, ds, args_dict):
    """"""
    if im_type == 'gim':
        return get_gim_impersonator(device=device, ckpt_path=ckpt_path, args_dict=args_dict)
    elif im_type == 'replay':
        return Impersonator(replay_impersonator)
    elif im_type == 'rnd_src':
        return Impersonator(lambda leaked_sample, n: rand_source_impersonator(leaked_sample, n, ds))
    else:
        assert False, "unsupported impersonator type"


def eval_game_for_pair(
        device,
        au_type, im_type, au_outdir, im_outdir,
        ds, batch_size, num_workers,
        ckpt_dir='ckpts', specific_model=None
):
    """"""
    # agents
    au_ckpt_path, au_args_dict = get_exp_args_from_dir(au_outdir, ckpt_dir, specific_model=specific_model)
    im_ckpt_path, im_args_dict = get_exp_args_from_dir(im_outdir, ckpt_dir, specific_model=specific_model)

    au_agent = get_authenticator(device=device, au_type=au_type, ckpt_path=au_ckpt_path, args_dict=au_args_dict)
    im_agent = get_impersonator(device=device, im_type=im_type, ckpt_path=im_ckpt_path, ds=ds, args_dict=im_args_dict)

    # evaluation
    acc, acc_on_fake, acc_on_real, auc = eval_authenticator_and_impersonator(
        device=device,
        ds=ds,
        batch_size=batch_size,
        num_workers=num_workers,
        authenticator=au_agent,
        impersonator=im_agent
    )

    return acc.item(), acc_on_fake.item(), acc_on_real.item(), auc.item()


def get_exp_args_from_dir(outdir, ckpt_dir, specific_model=None):
    """"""
    ckpt_dir_path = os.path.join(outdir, ckpt_dir)
    if specific_model is None:
        model_file_path = get_latest_ckpt(ckpt_dir_path)
    else:
        model_file_path = os.path.join(ckpt_dir_path, specific_model)
    args_dict = load_args(outdir)
    if not 'img_size' in args_dict:
        args_dict['img_size'] = args_dict['target_img_size']
    return model_file_path, args_dict


def eval_authentication_task(
        device,
        ds,
        m, n, k,
        batch_size, num_workers,
        gim_exp_dir,
        csv_file_path,
        specific_model=None,
        baseline_exp_dir=None,
        baseline_type=None,
):
    """"""
    if not os.path.isdir(os.path.dirname(csv_file_path)):
        os.makedirs(os.path.dirname(csv_file_path))

    cols = (
        'au_type','im_type',
        'ds_root', 'gim_exp_dir',
        'm','n','k',
        'acc', 'acc_on_fake', 'acc_on_real', 'auc'
    )
    printed_cols = ['au_type','im_type', 'acc', 'acc_on_fake', 'acc_on_real']
    df = pd.DataFrame(columns=cols)
    au_type_list = ['gim'] if (baseline_type is None) else ['gim', baseline_type]
    for au_type in au_type_list:
        for im_type in ('gim', 'replay', 'rnd_src'):
            print("running {} vs. {}".format(au_type, im_type))
            au_outdir = gim_exp_dir if au_type == 'gim' else baseline_exp_dir
            acc, acc_on_fake, acc_on_real, auc = eval_game_for_pair(
                device=device,
                au_type=au_type,
                im_type=im_type,
                au_outdir=au_outdir,
                im_outdir=gim_exp_dir,
                ds=ds, batch_size=batch_size, num_workers=num_workers,
                specific_model=specific_model
            )
            df_row = pd.DataFrame(
                [{
                    'au_type': au_type,
                    'im_type': im_type,
                    'ds_root': ds.root,
                    'gim_exp_dir': gim_exp_dir,
                    'm': m,
                    'n': n,
                    'k': k,
                    'acc': acc,
                    'acc_on_fake': acc_on_fake,
                    'acc_on_real': acc_on_real,
                    'auc': auc
                }]
            )
            print(df_row[printed_cols])
            df = df.append(df_row)

    # save to csv
    df.to_csv(csv_file_path)
    print(df[printed_cols])


def get_dataset(dataset_root,
                split,
                dataset_type,
                example_cnt_per_class,
                img_channels, img_size,
                m ,n ,k):
    # data
    if dataset_type == 'omniglot':
        ds = OmniglotGIMDataSet(
            root=dataset_root,
            split=split,
            img_channels=img_channels,
            img_size=img_size,
            m=m, n=n, si=k, example_cnt_per_class=example_cnt_per_class,
        )
    elif dataset_type == 'voxceleb2':
        ds = ImgGIMDataSet(
            root=dataset_root,
            split=split,
            img_channels=img_channels,
            img_size=img_size,
            m=m, n=n, si=k, example_cnt_per_class=example_cnt_per_class,
            hierarchical=True, mirror=True
        )
    elif dataset_type == 'general_imgs':
        ds = ImgGIMDataSet(
            root=dataset_root,
            split=split,
            img_channels=img_channels,
            img_size=img_size,
            m=m, n=n, si=k, example_cnt_per_class=example_cnt_per_class,
            hierarchical=False, mirror=True
        )
    else:
        raise ValueError("Supports only dataset_type in ['omniglot','voxceleb2','general_imgs']")
    return ds

########################################################################################################################
# App / Unit test
########################################################################################################################
def main(args):
    device = get_device(device_type=args.device, device_ids=args.device_ids)
    ds = get_dataset(
        dataset_root=args.ds_root,
        split=args.split,
        dataset_type=args.dataset_type,
        example_cnt_per_class=args.example_cnt_per_class,
        img_channels=args.img_channels,
        img_size=args.img_size,
        m=args.m, n=args.n, k=args.k
    )
    eval_authentication_task(
        device=device,
        ds=ds,
        m=args.m, n=args.n, k=args.k,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        baseline_exp_dir=args.baseline_exp_dir,
        baseline_type=args.baseline_type,
        gim_exp_dir=args.gim_exp_dir,
        csv_file_path=args.csv_file_path,
        specific_model=args.specific_model
    )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', default='cuda',
                        help='cuda or cpu')
    parser.add_argument('--device_ids', nargs='+', type=int, default=[0],
                        help='A list of device ids for the GPUS to be used. '
                             'E.g., if using GPUs 1,3,5,8, use: --device_ids 1 3 5 8. '
                             'Only relevant when using cuda.')
    parser.add_argument('--ds_root', required=True,
                        help='Path to dataset root dir.')
    parser.add_argument('--split', default = 'val',
                        help = 'train, val, or test')
    parser.add_argument('--dataset_type', default='omniglot',
                        help='omniglot or voxceleb2')
    parser.add_argument('--example_cnt_per_class', type=int, default=5,
                        help='How many examples to sample per class for the evaluation')
    parser.add_argument('--img_size', type=int, default=32,
                        help='image size')
    parser.add_argument('--img_channels', type=int, default=1, help='number of image channels')
    parser.add_argument('--m', type=int, default=1, help='m: the number of leaked images')
    parser.add_argument('--n', type=int, default=5, help='n: the number of test images')
    parser.add_argument('--k', type=int, default=5, help='k: the number of registration images')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--baseline_exp_dir', default=None,
                        help='experiment directory for the baseline model')
    parser.add_argument('--baseline_type',
                        default=None, help='siamese, arcface, or None')
    parser.add_argument('--gim_exp_dir', required=True,
                        help='experiment directory for the GIM model')
    parser.add_argument('--specific_model', default=None,
                        help='Path to a specific model checkpoint. If not specified, the latest model is taken.')
    parser.add_argument('--csv_file_path', default=os.path.join(os.path.abspath(os.path.dirname(__file__)),'results.csv'),
                        help='The path for the results csv file')
    return parser.parse_args()


# app
if __name__ == '__main__':
    args = get_args()
    main(args)
