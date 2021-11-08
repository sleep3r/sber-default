import fire
import pandas as pd


def merge(sub_path_1: str, sub_path_2: str, final_sub_path: str) -> None:
    sub1 = pd.read_csv(sub_path_1, sep=';')
    sub2 = pd.read_csv(sub_path_2, sep=';')

    final_sub = sub1.append(sub2, ignore_index=True).sort_values("id")
    final_sub.to_csv(final_sub_path, index=False, sep=';')


if __name__ == "__main__":
    fire.Fire(merge)
