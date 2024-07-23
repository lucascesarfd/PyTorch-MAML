import argparse
import os
import random
import sys

import git
import mlflow
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm 
from torch.utils.data import DataLoader

import datasets
import models
import utils
import utils.optimizers as optimizers


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

    # meta-train
    train_set = datasets.make(config["dataset"], **config["train"])
    data_channels = train_set[0][0].shape[1]
    data_in_dims = (train_set[0][0].shape[2], train_set[0][0].shape[3])
    params = {
        'train-shape': tuple(train_set[0][0].shape),
        'train-len': len(train_set),
        "train-classes": train_set.n_classes,
        }
    mlflow.log_params(params)
    print(f"Meta-Train set:\n\tshape: {tuple(train_set[0][0].shape)}\n\tlength: {len(train_set)}\n\tclasses: {train_set.n_classes}")
    train_loader = DataLoader(
        train_set,
        config["train"]["n_episode"],
        collate_fn=datasets.collate_fn,
        num_workers=config["custom"]["num_workers"],
        pin_memory=True,
    )

    # meta-val
    eval_val = False
    if config.get("val"):
        eval_val = True
        val_set = datasets.make(config["dataset"], **config["val"])
        params = {
            'val-shape': tuple(val_set[0][0].shape),
            'val-len': len(val_set),
            "val-classes": val_set.n_classes,
            }
        mlflow.log_params(params)
        print(f"Meta-Val set:\n\tshape: {tuple(val_set[0][0].shape)}\n\tlength: {len(val_set)}\n\tclasses: {val_set.n_classes}")
        val_loader = DataLoader(
            val_set,
            config["val"]["n_episode"],
            collate_fn=datasets.collate_fn,
            num_workers=config["custom"]["num_workers"],
            pin_memory=True,
        )

    ##### Model and Optimizer #####

    inner_args = utils.config_inner_args(config.get("inner_args"))
    if config.get("load"):
        ckpt = torch.load(config["load"])
        config["encoder"] = ckpt["encoder"]
        config["encoder_args"] = ckpt["encoder_args"]
        config["classifier"] = ckpt["classifier"]
        config["classifier_args"] = ckpt["classifier_args"]
        model = models.load(ckpt, load_clf=(not inner_args["reset_classifier"]))
        optimizer, lr_scheduler = optimizers.load(ckpt, model.parameters())
        start_epoch = ckpt["training"]["epoch"] + 1
        max_va = ckpt["training"]["max_va"]
    else:
        config["encoder_args"] = config.get("encoder_args") or dict()
        config["encoder_args"]["bn_args"]["n_episode"] = config["train"]["n_episode"]
        config["encoder_args"]["in_dims"] = data_in_dims
        config["encoder_args"]["channels"] = data_channels
        config["classifier_args"] = config.get("classifier_args") or dict()
        config["classifier_args"]["n_way"] = config["train"]["n_way"]
        model = models.make(
            config["encoder"],
            config["encoder_args"],
            config["classifier"],
            config["classifier_args"],
        )
        optimizer, lr_scheduler = optimizers.make(
            config["optimizer"], model.parameters(), **config["optimizer_args"]
        )
        start_epoch = 1
        max_va = 0.0

    if args.efficient:
        model.go_efficient()

    if config.get("_parallel"):
        model = nn.DataParallel(model)

    mlflow.log_param('num_params', utils.compute_n_params(model))
    print(f"Number of Parameters: {utils.compute_n_params(model)}")
    timer_elapsed, timer_epoch = utils.Timer(), utils.Timer()

    ##### Training and evaluation #####

    # 'tl': meta-train loss
    # 'ta': meta-train accuracy
    # 'vl': meta-val loss
    # 'va': meta-val accuracy
    aves_keys = ["tl", "ta", "vl", "va"]
    trlog = dict()
    for k in aves_keys:
        trlog[k] = []

    pbar = tqdm(
        range(start_epoch, config["epoch"] + 1),
        desc=f'Epoch',
        leave=False,
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"
    )

    for epoch in pbar:
        timer_epoch.start()
        aves = {k: utils.AverageMeter() for k in aves_keys}

        # meta-train
        model.train()
        mlflow.log_metric("lr", optimizer.param_groups[0]["lr"], step=epoch)
        np.random.seed(epoch)

        for data in tqdm(train_loader, desc="Meta-Train", leave=False, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"):
            x_shot, x_query, y_shot, y_query = data
            if config["custom"]["device"] != "cpu":
                x_shot, y_shot = x_shot.cuda(), y_shot.cuda()
                x_query, y_query = x_query.cuda(), y_query.cuda()

            if inner_args["reset_classifier"]:
                if config.get("_parallel"):
                    model.module.reset_classifier()
                else:
                    model.reset_classifier()

            logits = model(x_shot, x_query, y_shot, inner_args, meta_train=True)
            logits = logits.flatten(0, 1)
            labels = y_query.flatten()

            pred = torch.argmax(logits, dim=-1)
            acc = utils.compute_acc(pred, labels)
            loss = F.cross_entropy(logits, labels)
            aves["tl"].update(loss.item(), 1)
            aves["ta"].update(acc, 1)

            optimizer.zero_grad()
            loss.backward()
            for param in optimizer.param_groups[0]["params"]:
                nn.utils.clip_grad_value_(param, 10)
            optimizer.step()

        # meta-val
        if eval_val:
            model.eval()
            np.random.seed(0)

            for data in tqdm(val_loader, desc="Meta-Val", leave=False, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"):
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
                logits = logits.flatten(0, 1)
                labels = y_query.flatten()

                pred = torch.argmax(logits, dim=-1)
                acc = utils.compute_acc(pred, labels)
                loss = F.cross_entropy(logits, labels)
                aves["vl"].update(loss.item(), 1)
                aves["va"].update(acc, 1)

        if lr_scheduler is not None:
            lr_scheduler.step()

        for k, avg in aves.items():
            aves[k] = avg.item()
            trlog[k].append(aves[k])

        total_elapsed = timer_elapsed.end()
        total_estimate = total_elapsed / (epoch - start_epoch + 1) * (config["epoch"] - start_epoch + 1)
        estimate_time = total_estimate - total_elapsed

        # formats output
        metrics = {"loss-meta-train": aves["tl"], 'acc-meta-train': aves["ta"]}
        mlflow.log_metrics(metrics, step=epoch)

        if eval_val:
            metrics = {"loss-meta-val": aves["vl"], 'acc-meta-val': aves["va"]}
            mlflow.log_metrics(metrics, step=epoch)

        pbar.set_postfix(
            tr_loss=f'{aves["tl"]:.4f}',
            tr_acc=f'{aves["ta"]:.4f}',
            va_loss=f'{aves["vl"]:.4f}',
            va_acc=f'{aves["va"]:.4f}',
        )
        metrics = {"estimate_hour": estimate_time/3600, "epoch": epoch}
        mlflow.log_metrics(metrics, step=epoch)

        # saves model and meta-data
        if config.get("_parallel"):
            model_ = model.module
        else:
            model_ = model

        training = {
            "epoch": epoch,
            "max_va": max(max_va, aves["va"]),
            "optimizer": config["optimizer"],
            "optimizer_args": config["optimizer_args"],
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": (
                lr_scheduler.state_dict() if lr_scheduler is not None else None
            ),
        }
        ckpt = {
            "file": __file__,
            "config": config,
            "encoder": config["encoder"],
            "encoder_args": config["encoder_args"],
            "encoder_state_dict": model_.encoder.state_dict(),
            "classifier": config["classifier"],
            "classifier_args": config["classifier_args"],
            "classifier_state_dict": model_.classifier.state_dict(),
            "training": training,
        }

        # 'last.pth': saved at the latest epoch
        # 'best.pth': saved when validation accuracy is at its maximum
        last_path = os.path.join(ckpt_path, "last.pth")
        trlog_path = os.path.join(ckpt_path, "trlog.pth")
        torch.save(ckpt, last_path)
        torch.save(trlog, trlog_path)
        mlflow.log_artifact(os.path.abspath(last_path), artifact_path="model")
        mlflow.log_artifact(os.path.abspath(trlog_path), artifact_path="model")

        if aves["va"] > max_va:
            max_va = aves["va"]
            best_path = os.path.join(ckpt_path, "best.pth")
            torch.save(ckpt, best_path)
            mlflow.log_artifact(os.path.abspath(best_path), artifact_path="model")


if __name__ == "__main__":
    args = create_args()
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)

    # Deal with multiple GPU's
    if len(args.gpu.split(",")) > 1:
        config["_parallel"] = True
        config["_gpu"] = args.gpu

    utils.set_gpu(args.gpu)
    main(config)
