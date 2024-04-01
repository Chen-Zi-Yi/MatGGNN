#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project    : MatGGNN 
@File       : training.py
@IDE        : PyCharm 
@Author     : zychen@cnic.cn
@Date       : 2023/11/15 11:11 
@Description: 
"""
import copy
import csv
import os
import platform
import time

import numpy as np
import torch
import torch.distributed as dist
from sklearn.metrics import mean_absolute_error, f1_score
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler
from torch_geometric.loader import DataLoader

from Scripts import process
from matdeeplearn import models, model_summary, evaluate, train


def sigmoid(x):
    y = np.array(x)
    return 1/(1 + np.exp(-y))


##Write results to csv file
def write_results(output, filename):
    shape = output.shape
    with open(filename, "w") as f:
        csvwriter = csv.writer(f)
        for i in range(0, len(output)):
            if i == 0:
                csvwriter.writerow(
                    ["ids"]
                    + ["target"] * int((shape[1] - 1) / 2)
                    + ["prediction"] * int((shape[1] - 1) / 2)
                )
            elif i > 0:
                csvwriter.writerow(output[i - 1, :])



def ddp_setup(rank, world_size):
    if rank in ("cpu", "cuda"):
        return
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    if platform.system() == 'Windows':
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
    else:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = True


##Pytorch loader setup
def loader_setup(
        train_ratio,
        val_ratio,
        test_ratio,
        batch_size,
        dataset,
        rank,
        seed,
        world_size=0,
        num_workers=0,
):
    ##Split datasets
    train_dataset, val_dataset, test_dataset = process.split_data(
        dataset, train_ratio, val_ratio, test_ratio, seed
    )

    ##DDP
    if rank not in ("cpu", "cuda"):
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank
        )
    elif rank in ("cpu", "cuda"):
        train_sampler = None

    ##Load data
    train_loader = val_loader = test_loader = None
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        sampler=train_sampler,
    )
    # may scale down batch size if memory is an issue
    if rank in (0, "cpu", "cuda"):
        if len(val_dataset) > 0:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )
        if len(test_dataset) > 0:
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )
    return (
        train_loader,
        val_loader,
        test_loader,
        train_sampler,
        train_dataset,
        val_dataset,
        test_dataset,
    )


# pytorch model setup
def model_setup(
        rank,
        model_name,
        model_params,
        dataset,
        load_model=False,
        model_path=None,
        print_model=True
):
    model = getattr(models, model_name)(
        data=dataset, **(model_params if model_params is not None else {})
    ).to(rank)
    if load_model == "True":
        assert os.path.exists(model_path), "Saved model not found"
        if str(rank) in ("cpu"):
            saved = torch.load(model_path, map_location=torch.device("cpu"))
        else:
            saved = torch.load(model_path)
        model.load_state_dict(saved["model_state_dict"])
        # optimizer.load_state_dict(saved['optimizer_state_dict'])

    # DDP
    if rank not in ("cpu", "cuda"):
        model = DistributedDataParallel(
            model, device_ids=[rank], find_unused_parameters=True
        )
        # model = DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=False)
    if print_model and rank in (0, "cpu", "cuda"):
        model_summary(model)
    return model


def trainer(
        rank,
        world_size,
        model,
        optimizer,
        scheduler,
        loss,
        train_loader,
        val_loader,
        train_sampler,
        epochs,
        verbosity,
        filename="my_model_temp.pth",
):
    train_error = val_error = test_error = epoch_time = float("NaN")
    train_start = time.time()
    best_val_error = 1e10
    model_best = model
    ##Start training over epochs loop
    for epoch in range(1, epochs + 1):

        lr = scheduler.optimizer.param_groups[0]["lr"]
        if rank not in ("cpu", "cuda"):
            train_sampler.set_epoch(epoch)
        ## Train model
        train_error = train(model, optimizer, train_loader, loss, rank=rank)
        if rank not in ("cpu", "cuda"):
            torch.distributed.reduce(train_error, dst=0)
            train_error = train_error / world_size

        ##Get validation performance
        if rank not in ("cpu", "cuda"):
            dist.barrier()
        if val_loader != None and rank in (0, "cpu", "cuda"):
            if rank not in ("cpu", "cuda"):
                val_error = evaluate(
                    val_loader, model.module, loss, rank=rank, out=False
                )
            else:
                val_error = evaluate(val_loader, model, loss, rank=rank, out=False)

        ##Train loop timings
        epoch_time = time.time() - train_start
        train_start = time.time()

        ##remember the best val error and save model and checkpoint
        if val_loader != None and rank in (0, "cpu", "cuda"):
            if val_error == float("NaN") or val_error < best_val_error:
                if rank not in ("cpu", "cuda"):
                    model_best = copy.deepcopy(model.module)
                    torch.save(
                        {
                            "state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "full_model": model,
                        },
                        filename,
                    )
                else:
                    model_best = copy.deepcopy(model)
                    torch.save(
                        {
                            "state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "full_model": model,
                        },
                        filename,
                    )
            best_val_error = min(val_error, best_val_error)
        elif val_loader == None and rank in (0, "cpu", "cuda"):
            if rank not in ("cpu", "cuda"):
                model_best = copy.deepcopy(model.module)
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "full_model": model,
                    },
                    filename,
                )
            else:
                model_best = copy.deepcopy(model)
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "full_model": model,
                    },
                    filename,
                )

        ##scheduler on train error
        scheduler.step(train_error)

        ##Print performance
        if epoch % verbosity == 0:
            if rank in (0, "cpu", "cuda"):
                print(
                    "Epoch: {:04d}, Learning Rate: {:.6f}, Training Error: {:.5f}, Val Error: {:.5f}, Time per epoch (s): {:.5f}".format(
                        epoch, lr, train_error, val_error, epoch_time
                    )
                )

    if rank not in ("cpu", "cuda"):
        dist.barrier()

    return model_best


def training(rank, world_size, data_path, job_parameters=None, training_parameters=None, model_parameters=None,
             pretrained_model=None, filename='my_model_temp.pth', classification=False):
    ddp_setup(rank, world_size)
    if rank not in ('cpu', 'cuda'):
        model_parameters['lr'] = model_parameters['lr'] * world_size

    # Get Dataset
    dataset = process.get_dataset(data_path, training_parameters['target_index'], False)

    if rank not in ('cpu', 'cuda'):
        dist.barrier()

    # setup loader
    (
        train_loader,
        val_loader,
        test_loader,
        train_sampler,
        train_dataset,
        _,
        _,
    ) = loader_setup(
        training_parameters["train_ratio"],
        training_parameters["val_ratio"],
        training_parameters["test_ratio"],
        model_parameters["batch_size"],
        dataset,
        rank,
        job_parameters["seed"],
        world_size,
    )

    # setup model
    if pretrained_model is None:
        model = model_setup(rank, model_parameters['model'], model_parameters, dataset, job_parameters['load_model'],
                            job_parameters['model_path'], model_parameters.get('print_model', True))
        optimizer = getattr(torch.optim, model_parameters['optimizer'])(model.parameters(), lr=model_parameters['lr'],
                                                                        **model_parameters['optimizer_args'])
    else:
        model = pretrained_model.to(rank)
        optimizer = getattr(torch.optim, model_parameters["optimizer"])(
            filter(lambda p: p.requires_grad, model.parameters()), lr=model_parameters["lr"],
            **model_parameters["optimizer_args"])

    scheduler = getattr(torch.optim.lr_scheduler, model_parameters["scheduler"])(optimizer,
                                                                                 **model_parameters["scheduler_args"])
    # Start Training
    model = trainer(rank, world_size, model, optimizer, scheduler, training_parameters['loss'], train_loader,
                    val_loader, train_sampler, model_parameters['epochs'], training_parameters['verbosity'], filename)

    if rank in (0, "cpu", "cuda"):

        train_error = val_error = test_error = float("NaN")

        ##workaround to get training output in DDP mode
        ##outputs are slightly different, could be due to dropout or batchnorm?
        train_loader = DataLoader(
            train_dataset,
            batch_size=model_parameters["batch_size"],
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        ##Get train error in eval mode
        train_error, train_out = evaluate(
            train_loader, model, training_parameters["loss"], rank, out=True
        )
        print("Train Error: {:.5f}".format(train_error))

        ##Get val error
        if val_loader is not None:
            val_error, val_out = evaluate(
                val_loader, model, training_parameters["loss"], rank, out=True
            )
            true = [float(val_out[i][1]) for i in range(len(val_out))]
            pred = [float(val_out[i][2]) for i in range(len(val_out))]
            print("Val Error: {:.5f}".format(val_error))

        ##Get test error
        if test_loader is not None:
            test_error, test_out = evaluate(
                test_loader, model, training_parameters["loss"], rank, out=True
            )
            print("Test Error: {:.5f}".format(test_error));
            # true = [float(test_out[i][1]) for i in range(len(test_out))]
            # pred = [float(test_out[i][2]) for i in range(len(test_out))]
            # print("Test R2: {:.5f}".format(r2_score(true,pred)))

        ##Save model
        if job_parameters["save_model"]:

            if rank not in ("cpu", "cuda"):
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "full_model": model,
                    },
                    job_parameters["model_path"],
                )
            else:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "full_model": model,
                    },
                    job_parameters["model_path"],
                )

        ##Write outputs
        if job_parameters["write_output"]:

            write_results(
                train_out, str(job_parameters["job_name"]) + "_train_outputs.csv"
            )
            if val_loader is not None:
                write_results(
                    val_out, str(job_parameters["job_name"]) + "_val_outputs.csv"
                )
            if test_loader is not None:
                write_results(
                    test_out, str(job_parameters["job_name"]) + "_test_outputs.csv"
                )

        if rank not in ("cpu", "cuda"):
            dist.destroy_process_group()

        ##Write out model performance to file
        error_values = np.array((train_error.cpu(), val_error.cpu(), test_error.cpu()))
        if job_parameters.get("write_error") == "True":
            np.savetxt(
                job_parameters["job_name"] + "_errorvalues.csv",
                error_values[np.newaxis, ...],
                delimiter=",",
            )

        if pretrained_model is not None and not classification:
            error_values = [error_values, mean_absolute_error(true, pred)]
        elif pretrained_model is not None and classification:
            error_values = [error_values, f1_score(np.array(true).astype('int'), np.round(sigmoid(pred)).astype('int'))]

        return error_values
