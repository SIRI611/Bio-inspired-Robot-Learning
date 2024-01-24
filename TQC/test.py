import dill

with open("trace_continue_train/walker2d_tqc_trace_continue_train.pkl", "rb") as f:
    data = dill.load(f)

print(data["mu_bias_amp"])