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
import csv


def eval():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--model', '-m', type=str, required=True)
    parser.add_argument('--inf', type=int, default=10)
    parser.add_argument('--outf', type=int, default=10)
    parser.add_argument('--test_data_index', type=int, nargs='+', default=[4000, 4416])
    parser.add_argument('--eval_data_range_index', type=int, nargs='+', default=[0, 100])
    parser.add_argument('--files', '-f', nargs='+', default=['jmagpz.npy'])
    args = parser.parse_args()

    test = dataset.JmaGpzDataset(args.test_data_index[0], args.test_data_index[1],
                                 args.inf, args.outf, file=args.files)

    model = network.JmaGpzNetwork(sz=[128, 64, 64], n=1, directory="img/")

    print("loading model from " + args.model)
    serializers.load_npz(args.model, model)

    all_results = [[]] * 3
    for i in range(3):
        all_results[i] = []

    for target_frame in range(args.outf):
        results = []
        for i in range(args.eval_data_range_index[0], args.eval_data_range_index[1]):
            x, t = test[i]

            x = np.expand_dims(x, 0)
            t = np.expand_dims(t, 0)

            if args.gpu >= 0:
                cuda.get_device_from_id(0).use()
                model.to_gpu()
                t = cuda.cupy.array(t)
            score = model.eval(x, t, target_frame)
            if score is not None:
                results.append(score)

        results = np.array(results)
        print(f'target_frame: {target_frame}')
        print(f'N: {len(results)}')
        print(f'average: {np.average(results, axis=0)}')
        print(f'median: {np.median(results, axis=0)}')
        print(f'std: {np.std(results, axis=0)}')
        avg = np.average(results, axis=0)
        all_results[target_frame] = avg

    with open("eval.csv","w") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(['csi', 'far', 'pod'])
        for result in all_results:
            writer.writerow(result)


if __name__ == '__main__':
    eval()

