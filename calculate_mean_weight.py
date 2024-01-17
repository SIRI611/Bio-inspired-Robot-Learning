import numpy as np
import dill
import torch

if __name__ == "__main__":
    with open("trace_gradient.pkl", "rb") as f:
        load_data = dill.load(f)
    # print(len(gradient["weight"]))
    # print(len(gradient["name"]))
    print(len(load_data["weight"]))

    gradient_mean = [load_data["weight"][-6],
                     load_data["weight"][-5],
                     load_data["weight"][-4],
                     load_data["weight"][-3],
                     load_data["weight"][-2],
                     load_data["weight"][-1]]
    
    with open('weight_mean.pkl', 'wb') as f:
        dill.dump(gradient_mean, f) 