#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import argparse
import chainer
from chainer import serializers
from chainer import Variable
from chainer import cuda
import dataset
import network
from PIL import Image

def generate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--model', '-m', type=str, default=None)
    parser.add_argument('--id', '-i', type=int, default=0)
    parser.add_argument('--inf', type=int, default=10)
    parser.add_argument('--outf', type=int, default=10)
    args = parser.parse_args()

    test = dataset.JmaGpzDataset(0, 4416, args.inf, args.outf)
    model = network.JmaGpzNetwork(sz=[128, 64, 64], n=1, directory="img/")

    if args.model != None:
        print( "loading model from " + args.model )
        serializers.load_npz(args.model, model)

    x, t = test[args.id]

    x = np.expand_dims(x, 0)
    t = np.expand_dims(t, 0)


    if args.gpu >= 0:
        cuda.get_device_from_id(0).use()
        model.to_gpu()
        t = cuda.cupy.array(t)

    res = model(x, t)

if __name__ == '__main__':
    generate()
