# imports
import os
import sys
import itertools
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score


# local imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(project_root)


########################################################################################################################
# Functions
########################################################################################################################
def write_results(file_path, acc, acc_on_fake, acc_on_real, print_to_stdout=False):
    """"""
    s = "accuracy: {}\naccuracy on fake: {}\naccuracy on real: {}\n".format(acc, acc_on_fake, acc_on_real)
    file_dir = os.path.dirname(file_path)
    if not os.path.isdir(file_dir):
        os.makedirs(file_dir)
    with open(file_path, 'w') as f:
        f.write(s)
    if print_to_stdout:
        print(s)


def comp_acc(pred_on_real, pred_on_fake):
    """
    :param pred_on_real:
    :param pred_on_fake:
    :return:
    """
    assert len(pred_on_real.size()) == 1 and len(pred_on_fake.size()) == 1
    assert pred_on_real.size(0) == pred_on_fake.size(0)
    acc_on_real = pred_on_real.to(torch.float).mean()
    acc_on_fake = torch.eq(pred_on_fake, 0).to(torch.float).mean()
    acc = 0.5 * (acc_on_real + acc_on_fake)
    return acc, acc_on_fake, acc_on_real


def eval_authenticator_and_impersonator(
        device,
        ds,
        batch_size,
        num_workers,
        authenticator,
        impersonator,
        dbg=False
):
    """"""
    pred_on_fake_list = []
    pred_on_real_list = []
    out_on_fake_list = []
    out_on_real_list = []

    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    num_iters = 1000 if dbg else len(dataloader)
    iter_bar = tqdm(itertools.islice(dataloader, num_iters), total=num_iters, desc='Eval Authentication')
    for batch_idx, data_batch in enumerate(iter_bar):
        # data
        real_sample = data_batch["real_sample"].to(device)
        leaked_sample = data_batch["leaked_sample"].to(device)
        si_sample = data_batch["si_sample"].to(device)
        n = real_sample.size(1)

        # predictions
        out_on_real, pred_on_real = authenticator.act(
            test_sample=real_sample, si_sample=si_sample,
        )
        fake_sample = impersonator.act(leaked_sample=leaked_sample, n=n)
        out_on_fake, pred_on_fake = authenticator.act(
            test_sample=fake_sample, si_sample=si_sample,
        )

        # store:
        out_on_real_list.append(out_on_real.view(-1).detach())
        out_on_fake_list.append(out_on_fake.view(-1).detach())
        pred_on_real_list.append(pred_on_real.view(-1).detach())
        pred_on_fake_list.append(pred_on_fake.view(-1).detach())

    out_on_real = torch.cat(out_on_real_list)
    out_on_fake = torch.cat(out_on_fake_list)
    pred_on_real = torch.cat(pred_on_real_list)
    pred_on_fake = torch.cat(pred_on_fake_list)

    # comp acc
    acc, acc_on_fake, acc_on_real = comp_acc(pred_on_real=pred_on_real, pred_on_fake=pred_on_fake)

    # comp auc
    y_true = torch.cat([torch.ones_like(out_on_real), torch.zeros_like(out_on_fake)]).cpu().numpy()
    y_score = torch.cat([out_on_real, out_on_fake]).cpu().numpy()
    auc = roc_auc_score(y_true=y_true, y_score=y_score)
    return acc, acc_on_fake, acc_on_real, auc


def eval_dis_on_multiple_im(
        device,
        ds,
        batch_size,
        num_workers,
        authenticator,
        impersonator_dict,
):
    """"""
    results = {}
    for im_key in impersonator_dict.keys():
        print("\nEvaluating on impersonator: {}\n".format(im_key))
        acc, acc_on_fake, acc_on_real, auc = eval_authenticator_and_impersonator(
            device=device,
            ds=ds,
            batch_size=batch_size,
            num_workers=num_workers,
            authenticator=authenticator,
            impersonator=impersonator_dict[im_key]
        )
        results[im_key] = {"acc": acc, "acc_on_fake": acc_on_fake, "acc_on_real": acc_on_real, "auc": auc}

    return results