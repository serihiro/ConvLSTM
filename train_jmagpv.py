#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import argparse
import chainer
from chainer import training
from chainer import iterators, optimizers, serializers
from chainer import cuda
from chainer.training import extensions
import dataset
import network


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, required=True)
    parser.add_argument('--val', type=str, required=True)
    parser.add_argument('--n_in', type=int, default=6)
    parser.add_argument('--n_out', type=int, default=6)
    parser.add_argument('--iteration', type=int, default=5000)
    parser.add_argument('--snapshot_interval', type=int, default=1000)
    parser.add_argument('--batch_size', '-b', type=int, default=1)
    parser.add_argument('--n_process', type=int, default=1)
    parser.add_argument('--n_prefetch', type=int, default=1)
    parser.add_argument('--model', '-m', type=str, default=None)
    parser.add_argument('--out', type=str, default='result')
    parser.add_argument('--lr', '-l', type=float, default=0.001)
    args = parser.parse_args()

    train = dataset.JmaGpvDataset(index_file_path=args.train, n_in=args.n_in, n_out=args.n_out)
    train_iter = chainer.iterators.MultiprocessIterator(
        dataset=train,
        batch_size=args.batch_size,
        n_processes=args.n_process,
        n_prefetch=args.n_prefetch
    )
    val = dataset.JmaGpvDataset(index_file_path=args.val, n_in=args.n_in, n_out=args.n_out)
    val_iter = chainer.iterators.MultiprocessIterator(
        dataset=val,
        batch_size=args.batch_size,
        repeat=False,
        n_processes=args.n_process,
        n_prefetch=args.n_prefetch
    )

    model = network.MovingMnistNetwork(sz=[128, 64, 64], n=2)

    if args.model is not None:
        print("loading model from " + args.model)
        serializers.load_npz(args.model, model)

    cuda.get_device_from_id(0).use()
    model.to_gpu()

    opt = optimizers.Adam(alpha=args.lr)
    opt.setup(model)

    updater = training.StandardUpdater(train_iter, opt, device=0)

    trainer = training.Trainer(updater, (args.iteration, 'iteration'), out=args.out)
    trainer.extend(extensions.Evaluator(val_iter, model, device=0))
    trainer.extend(extensions.LogReport(trigger=(10, 'iteration')))
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss']))
    trainer.extend(
        extensions.snapshot_object(
            model,
            'model_iter_{.updater.iteration}'
        ),
        trigger=(args.snapshot_interval, 'iteration')
    )
    trainer.extend(extensions.ProgressBar(update_interval=1))

    trainer.run()


if __name__ == '__main__':
    train()
