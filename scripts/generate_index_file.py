import argparse
import glob
import os
import concurrent.futures


def main(root: str, output: str, concurrency: int):
    all_file_path = glob.glob(f'{root}/**/*.npy')
    all_file_full_path = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=concurrency) as executor:
        all_file_full_path = list(executor.map(os.path.abspath, all_file_path))

    with open(output, mode='w') as f:
        f.write('\n'.join(all_file_full_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--concurrency', type=int, default=1)
    args = vars(parser.parse_args())
    main(**args)
