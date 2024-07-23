import argparse
import os
import random
import sys

import git
import mlflow
import yaml
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

import datasets
import models
import utils


def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="configuration file")
    parser.add_argument("--log_dir", help="the log directory", type=str, default="./save")
    parser.add_argument("--gpu", help="gpu device number", type=str, default="0")
    parser.add_argument("--efficient", help="if True, enables gradient checkpointing", action="store_true")
    args = parser.parse_args()
   
    return args


def main(config):
    # Set random seeds
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    ##### MLFLOW #####

    # set mlflow tracking folder
    mlflow.set_tracking_uri(config["mlflow"]["uri"])

    # set experiment name
    mlflow.set_experiment(config["mlflow"]["exp_name"])

    # if a run with the parameter run_id exists, log to that experiment
    ml_run = mlflow.start_run(run_id=config["mlflow"]["run_id"], run_name=config["mlflow"]["run_name"])

    # register git commit for experiment reproducibility
    if config["mlflow"]["git_repo"]:
        git_repo = git.Repo(config["mlflow"]["git_repo"])
        mlflow.set_tag('branch',
                    None if git_repo.head.is_detached else git_repo.active_branch.name)
        mlflow.set_tag('commit', git_repo.head.commit)
    
    # register full bash command as tags
    mlflow.set_tag('full_command', ' '.join(sys.argv))

    # Deal with logs dir
    mlflow.log_artifact(os.path.abspath(args.config))    
    
    ckpt_path = os.path.join(args.log_dir, f'{config["mlflow"]["exp_name"]}', ml_run.info.run_id)
    utils.ensure_path(ckpt_path)

    ##### Dataset #####

    dataset = datasets.make(config["dataset"], **config["test"])
    params = {
        'test-shape': tuple(dataset[0][0].shape),
        'test-len': len(dataset),
        "test-classes": dataset.n_classes,
        }
    mlflow.log_params(params)
    print(f"Meta-Test set:\n\tshape: {tuple(dataset[0][0].shape)}\n\tlength: {len(dataset)}\n\tclasses: {dataset.n_classes}")

    loader = DataLoader(
        dataset,
        config["test"]["n_episode"],
        collate_fn=datasets.collate_fn,
        num_workers=config["custom"]["num_workers"],
        pin_memory=True,
    )

    ##### Model #####

    ckpt = torch.load(config["load"], map_location="cpu")
    inner_args = utils.config_inner_args(config.get("inner_args"))
    model = models.load(ckpt, load_clf=(not inner_args["reset_classifier"]))

    if args.efficient:
        model.go_efficient()

    if config.get("_parallel"):
        model = nn.DataParallel(model)

    mlflow.log_param('num_params', utils.compute_n_params(model))
    print(f"Number of Parameters: {utils.compute_n_params(model)}")

    ##### Evaluation #####

    model.eval()
    aves_va = utils.AverageMeter()
    va_lst = []

    for epoch in range(1, config["epoch"] + 1):
        for data in tqdm(loader, leave=False, desc="Meta-Test"):
            x_shot, x_query, y_shot, y_query = data
            if config["custom"]["device"] != "cpu":
                x_shot, y_shot = x_shot.cuda(), y_shot.cuda()
                x_query, y_query = x_query.cuda(), y_query.cuda()

            if inner_args["reset_classifier"]:
                if config.get("_parallel"):
                    model.module.reset_classifier()
                else:
                    model.reset_classifier()

            logits = model(x_shot, x_query, y_shot, inner_args, meta_train=False)
            logits = logits.view(-1, config["test"]["n_way"])
            labels = y_query.view(-1)

            pred = torch.argmax(logits, dim=1)
            acc = utils.compute_acc(pred, labels)
            aves_va.update(acc, 1)
            va_lst.append(acc)
        
        metrics = {"acc-test-mean": aves_va.item(), "acc-test-std": utils.mean_confidence_interval(va_lst)}
        mlflow.log_metrics(metrics, step=epoch)

        print(
            "Test epoch {}: ACC={:.2f} +- {:.2f} (%)".format(
                epoch, aves_va.item() * 100, utils.mean_confidence_interval(va_lst) * 100
            )
        )


if __name__ == "__main__":
    args = create_args()
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)

    if len(args.gpu.split(",")) > 1:
        config["_parallel"] = True
        config["_gpu"] = args.gpu

    utils.set_gpu(args.gpu)
    main(config)
