import argparse
import os

import polars as pl

TRAIN_CSV_PATH = "AliExpress_NL/test.csv"
TEST_CSV_PATH = "AliExpress_NL/test.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--fraction", type=float, default=0.1)
    args = parser.parse_args()

    fraction = args.fraction

    df = pl.read_csv(TRAIN_CSV_PATH)

    df_click = df.filter(pl.col("click") == 1)

    df_nonclick = df.filter(pl.col("click") == 0)
    df_nonclick_sampled = df_nonclick.sample(fraction=fraction)

    df_sampled = pl.concat([df_click, df_nonclick_sampled])
    df_sampled.write_csv(f"train_sampled_{fraction}.csv")


if __name__ == "__main__":
    main()
