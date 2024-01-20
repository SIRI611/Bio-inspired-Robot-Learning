#### Stable Baselines 3 pre-train DRL Models

Our repo implements three DRL algorisms: SAC, TD3 and TQC, we use these three algorisms  train Mujoco tasks: Humanoid-v4, Hopper-v4, HalfCheetah-v4, Ant-v4, and Walker2d-v4

How  to run our code:

Please run the python file named **{}_main.py** in each floder. The Config class set the parameters. Take td3_humanoid as an example. The Config file is as follows:

```python
class Config():
    def __init__(self) -> None:
        self.total_step = 2e6
        self.is_train = True
        self.is_continue_train = False
        self.env = "Humanoid-v4"
        self.dt = 15
        self.num_test = 10
        self.env_name="humanoid_td3"
        self.logpath = "td3_humanoid_tensorboard"
```

Switching tasks requires changing the "env" parameter. Please note that after changing the parameters, update the corresponding logpath and env_name parameters. Also, configure different hyperparameters for each task. Refer to the hyperparameter folder for guidance.