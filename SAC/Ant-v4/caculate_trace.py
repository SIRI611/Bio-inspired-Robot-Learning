import numpy as np
import dill
import torch

if __name__ == "__main__":
    with open("trace.pkl", "rb") as f:
        load_data = dill.load(f)
    # print(len(gradient["gradient"]))
    # print(len(gradient["name"]))
    # print(load_data["gradient"])

    print(load_data["name"])