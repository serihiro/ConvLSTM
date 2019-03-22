#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import argparse

import numpy as np
from chainer import Variable
from chainer import cuda
from chainer import serializers

import dataset
import network


def generate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--n_in', type=int, default=6)
    parser.add_argument('--n_out', type=int, default=6)
    parser.add_argument('--out', type=str, default="img_jmagpv")
    parser.add_argument('--id', type=int, default=0)
    args = parser.parse_args()

    test = dataset.JmaGpvDataset(index_file_path=args.test, n_in=args.n_in, n_out=args.n_out)
    model = network.MovingMnistNetwork(sz=[128, 64, 64], n=2, directory=args.out)

    print("loading model from " + args.model)
    serializers.load_npz(args.model, model)

    x, t = test[args.id]

    x = np.expand_dims(x, 0)
    t = np.expand_dims(t, 0)

    cuda.get_device_from_id(0).use()
    model.to_gpu()
    x = cuda.cupy.array(x)
    t = cuda.cupy.array(t)

    res = model(Variable(x), Variable(t))


if __name__ == '__main__':
    generate()
