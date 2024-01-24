import dill
import numpy as np
import torch

class Config():
    def __init__(self) -> None:
        self.algothrim = "TQC" #"TD3", "TQC"
        self.env_name = "humanoid_tqc"
        self.trace_name = "trace/trace_humanoid_tqc_2000000.0.pkl"
        self.mode = "mean" #"mean", "abs_mean"
        self.trace_num = 600

def calculate_mean_gradient(load_data, trace_num, savepath):

    layer_intermediate_0_weight = [load_data["gradient"][8*i] for i in range(trace_num)]
    layer_intermediate_0_bias = [load_data["gradient"][8*i+1] for i in range(trace_num)]
    layer_intermediate_1_weight = [load_data["gradient"][8*i+2] for i in range(trace_num)]
    layer_intermediate_1_bias = [load_data["gradient"][8*i+3] for i in range(trace_num)]
    mu_layer_weight = [load_data["gradient"][8*i+4] for i in range(trace_num)]
    mu_layer_bias = [load_data["gradient"][8*i+5] for i in range(trace_num)]
    log_std_layer_weight = [load_data["gradient"][8*i+6] for i in range(trace_num)]
    log_std_layer_bias = [load_data["gradient"][8*i+7] for i in range(trace_num)]

    layer_intermediate_0_weight_average = torch.mean(torch.stack(layer_intermediate_0_weight), dim=0)
    layer_intermediate_0_bias_average = torch.mean(torch.stack(layer_intermediate_0_bias), dim=0)
    layer_intermediate_1_weight_average = torch.mean(torch.stack(layer_intermediate_1_weight), dim=0)
    layer_intermediate_1_bias_average = torch.mean(torch.stack(layer_intermediate_1_bias), dim=0)
    mu_layer_weight_average = torch.mean(torch.stack(mu_layer_weight), dim=0)
    mu_layer_bias_average = torch.mean(torch.stack(mu_layer_bias), dim=0)
    log_std_layer_weight_average = torch.mean(torch.stack(log_std_layer_weight), dim=0)
    log_std_layer_bias_average = torch.mean(torch.stack(log_std_layer_bias), dim=0)


    gradient_mean = [layer_intermediate_0_weight_average, 
                     layer_intermediate_0_bias_average, 
                     layer_intermediate_1_weight_average, 
                     layer_intermediate_1_bias_average,
                     mu_layer_weight_average,
                     mu_layer_bias_average,
                     log_std_layer_weight_average,
                     log_std_layer_bias_average]
    
    with open(savepath, 'wb') as f:
        dill.dump(gradient_mean, f) 


def calculate_max_gradient(load_data, trace_num, savepath):

    layer_intermediate_0_weight = [load_data["gradient"][8*i] for i in range(trace_num)]
    layer_intermediate_0_bias = [load_data["gradient"][8*i+1] for i in range(trace_num)]
    layer_intermediate_1_weight = [load_data["gradient"][8*i+2] for i in range(trace_num)]
    layer_intermediate_1_bias = [load_data["gradient"][8*i+3] for i in range(trace_num)]
    mu_layer_weight = [load_data["gradient"][8*i+4] for i in range(trace_num)]
    mu_layer_bias = [load_data["gradient"][8*i+5] for i in range(trace_num)]
    log_std_layer_weight = [load_data["gradient"][8*i+6] for i in range(trace_num)]
    log_std_layer_bias = [load_data["gradient"][8*i+7] for i in range(trace_num)]

    layer_intermediate_0_weight_average = torch.max(torch.stack(layer_intermediate_0_weight), dim=0)
    layer_intermediate_0_bias_average = torch.max(torch.stack(layer_intermediate_0_bias), dim=0)
    layer_intermediate_1_weight_average = torch.max(torch.stack(layer_intermediate_1_weight), dim=0)
    layer_intermediate_1_bias_average = torch.max(torch.stack(layer_intermediate_1_bias), dim=0)
    mu_layer_weight_average = torch.max(torch.stack(mu_layer_weight), dim=0)
    mu_layer_bias_average = torch.max(torch.stack(mu_layer_bias), dim=0)
    log_std_layer_weight_average = torch.max(torch.stack(log_std_layer_weight), dim=0)
    log_std_layer_bias_average = torch.max(torch.stack(log_std_layer_bias), dim=0)


    gradient_max = [layer_intermediate_0_weight_average.values, 
                     layer_intermediate_0_bias_average.values, 
                     layer_intermediate_1_weight_average.values, 
                     layer_intermediate_1_bias_average.values,
                     mu_layer_weight_average.values,
                     mu_layer_bias_average.values,
                     log_std_layer_weight_average.values,
                     log_std_layer_bias_average.values]
    
    with open(savepath, 'wb') as f:
        dill.dump(gradient_max, f) 

def calculate_abs_mean_gradient(load_data, trace_num, savepath):

    layer_intermediate_0_weight = [torch.abs(load_data["gradient"][8*i]) for i in range(trace_num)]
    layer_intermediate_0_bias = [torch.abs(load_data["gradient"][8*i+1]) for i in range(trace_num)]
    layer_intermediate_1_weight = [torch.abs(load_data["gradient"][8*i+2]) for i in range(trace_num)]
    layer_intermediate_1_bias = [torch.abs(load_data["gradient"][8*i+3]) for i in range(trace_num)]
    mu_layer_weight = [torch.abs(load_data["gradient"][8*i+4]) for i in range(trace_num)]
    mu_layer_bias = [torch.abs(load_data["gradient"][8*i+5]) for i in range(trace_num)]
    log_std_layer_weight = [torch.abs(load_data["gradient"][8*i+6]) for i in range(trace_num)]
    log_std_layer_bias = [torch.abs(load_data["gradient"][8*i+7]) for i in range(trace_num)]


    layer_intermediate_0_weight_average = torch.mean(torch.stack(layer_intermediate_0_weight), dim=0)
    layer_intermediate_0_bias_average = torch.mean(torch.stack(layer_intermediate_0_bias), dim=0)
    layer_intermediate_1_weight_average = torch.mean(torch.stack(layer_intermediate_1_weight), dim=0)
    layer_intermediate_1_bias_average = torch.mean(torch.stack(layer_intermediate_1_bias), dim=0)
    mu_layer_weight_average = torch.mean(torch.stack(mu_layer_weight), dim=0)
    mu_layer_bias_average = torch.mean(torch.stack(mu_layer_bias), dim=0)
    log_std_layer_weight_average = torch.mean(torch.stack(log_std_layer_weight), dim=0)
    log_std_layer_bias_average = torch.mean(torch.stack(log_std_layer_bias), dim=0)

    gradient_max = [layer_intermediate_0_weight_average, 
                     layer_intermediate_0_bias_average, 
                     layer_intermediate_1_weight_average, 
                     layer_intermediate_1_bias_average,
                     mu_layer_weight_average,
                     mu_layer_bias_average,
                     log_std_layer_weight_average,
                     log_std_layer_bias_average]
    
    with open(savepath, 'wb') as f:
        dill.dump(gradient_max, f) 

def record_weight(load_data, savepath):

    weight = [load_data["weight"][-8],
        load_data["weight"][-7],
        load_data["weight"][-6],
        load_data["weight"][-5],
        load_data["weight"][-4],
        load_data["weight"][-3],
        load_data["weight"][-2],
        load_data["weight"][-1]]
    with open(savepath, 'wb') as f:
        dill.dump(weight, f) 
    
if __name__ == "__main__":
    para = Config()
    with open(para.trace_name, "rb") as f:
        load_data = dill.load(f)
    print(load_data["name"])
    if para.mode == "mean":
        savepath_gradient = "save_gradient/{}_{}_gradient_{}.pkl".format(para.env_name, para.mode, para.trace_num)
        calculate_mean_gradient(load_data=load_data, trace_num=para.trace_num, savepath=savepath_gradient)
    
    if para.mode == "max":
        savepath_gradient = "save_gradient/{}_{}_gradient_{}.pkl".format(para.env_name, para.mode, para.trace_num)
        calculate_max_gradient(load_data=load_data, trace_num=para.trace_num, savepath=savepath_gradient)

    if para.mode == "abs_mean":
        savepath_gradient = "save_gradient/{}_{}_gradient_{}.pkl".format(para.env_name, para.mode, para.trace_num)
        calculate_abs_mean_gradient(load_data=load_data, trace_num=para.trace_num, savepath=savepath_gradient)
    
    savepath_weight = "save_weight/{}_weight.pkl".format(para.env_name)
    record_weight(load_data=load_data, savepath=savepath_weight)
    print(load_data["name"])
    