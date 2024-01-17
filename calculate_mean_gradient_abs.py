import numpy as np
import dill
import torch

if __name__ == "__main__":
    with open("trace_gradient.pkl", "rb") as f:
        load_data = dill.load(f)
    # print(len(gradient["gradient"]))
    # print(len(gradient["name"]))
    # print(load_data["gradient"])

    layer_intermediate_0_weight = [torch.abs(load_data["gradient"][6*i]) for i in range(1000)]
    layer_intermediate_0_bias = [torch.abs(load_data["gradient"][6*i+1]) for i in range(1000)]
    layer_intermediate_1_weight = [torch.abs(load_data["gradient"][6*i+2]) for i in range(1000)]
    layer_intermediate_1_bias = [torch.abs(load_data["gradient"][6*i+3]) for i in range(1000)]
    mu_log_std_layer_weight = [torch.abs(load_data["gradient"][6*i+4]) for i in range(1000)]
    mu_log_std_layer_bias = [torch.abs(load_data["gradient"][6*i+5]) for i in range(1000)]


    layer_intermediate_0_weight_average = torch.mean(torch.stack(layer_intermediate_0_weight), dim=0)
    layer_intermediate_0_bias_average = torch.mean(torch.stack(layer_intermediate_0_bias), dim=0)
    layer_intermediate_1_weight_average = torch.mean(torch.stack(layer_intermediate_1_weight), dim=0)
    layer_intermediate_1_bias_average = torch.mean(torch.stack(layer_intermediate_1_bias), dim=0)
    mu_log_std_layer_weight_average = torch.mean(torch.stack(mu_log_std_layer_weight), dim=0)
    mu_log_std_layer_bias_average = torch.mean(torch.stack(mu_log_std_layer_bias), dim=0)

    gradient_mean = [layer_intermediate_0_weight_average, 
                     layer_intermediate_0_bias_average, 
                     layer_intermediate_1_weight_average, 
                     layer_intermediate_1_bias_average,
                     mu_log_std_layer_weight_average,
                     mu_log_std_layer_bias_average]
    
    with open('gradient_mean_abs.pkl', 'wb') as f:
        dill.dump(gradient_mean, f) 