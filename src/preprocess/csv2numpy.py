import argparse
import os
from tqdm import tqdm
import numpy as np
import pandas as pd

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=str, help="Path to intensities csv file",
                        default='inputs_bin1')
    parser.add_argument("--g", type=str, help="Time of run",
                        default='30')
    parser.add_argument("--m", type=str, help="m",
                        default='2')
    parser.add_argument("--set", type=str, help="train/valid",
                        default='train')
    args = parser.parse_args()

    inputs_dir = f"{args.inputs}/g{args.g}/m{args.m}/{args.set}"

    new_dir = inputs_dir + '_numpy'

    os.makedirs(new_dir, exist_ok=True)
    files = os.listdir(inputs_dir)

    pbar = tqdm(total=len(files))
    for i, file in enumerate(files):
        new_file = '.'.join([file.split(".csv")[0], 'pickle'])
        arr = pd.read_csv(f"{inputs_dir}/{file}")
        with open(f"{inputs_dir}_numpy/{new_file}", 'wb') as f:
            np.save(f, arr.to_numpy())
        if i == 0:
            with open(f"{inputs_dir}/../columns", 'wb') as f:
                np.save(f, arr.columns)
        pbar.update(1)
        del arr