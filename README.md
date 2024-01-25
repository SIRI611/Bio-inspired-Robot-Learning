## Stable Baselines 3 pre-train DRL Models

Our repo implements three DRL algorisms: **SAC, TD3 and TQC**, we use these three algorisms  train Mujoco tasks: **Humanoid-v4, Hopper-v4, HalfCheetah-v4, Ant-v4, and Walker2d-v4**


### Instructions

#### Pre-train model with stable-baseline3

Let’s take TD3 model and Ant-v4 task for an example!

 First, configure parameters. In the file **TD3/td3_main.py**, set **Config.is_train** as True and **Config.is_continue_train** as False. Meanwhile, configure parameters like env, env_name, and logpath. The Config class in the python file is as follows:

```python
class Config():
    def __init__(self) -> None:
        self.total_step = 2e6
        self.is_train = True
        self.is_continue_train = False
        self.env = "Ant-v4"
        self.dt = 15
        self.num_test = 10
        self.env_name="ant_td3"
        self.logpath = "tensorboard/td3_ant_tensorboard"
```

Next, run td3_main.py:

```shell
cd TD3/
python td3_main.py
```

#### Calculate gradient and weight

Firstly, please confirm that the trace file has been saved in the corresponding algorithm's "trace" folder.

Then configure the parameters in the **calculate_trace.py**, the Config class is as follows:

```python
class Config():
    def __init__(self) -> None:
        self.algothrim = "TD3" #"SAC", "TQC"
        self.env_name = "ant_td3"
        self.trace_name = "trace/trace_ant_td3_1000000.pkl"
        self.mode = "max" #"mean", "abs_mean"
        self.trace_num = 600
```

Next, run calculate_trace.py:

```shell
cd TD3/
python calculate_trace.py
```

#### Train pre-train model with dynamic synapse

Configure the parameters, in the td3_main.py, set **Config.is_train** as False and **Config.is_continue_train** as True. and then run the td3_main.py:

```shell
cd TD3/
python td3_main.py
```
