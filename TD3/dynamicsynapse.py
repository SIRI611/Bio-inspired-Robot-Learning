import copy
import math
import sys

# from DynamicSynapse.Adapter.RangeAdapter import RangeAdapter
import numpy as np
import torch
from torch.optim.optimizer import Optimizer

def f(t, period):
    t = t % period
    value = torch.where(t <= period / 4, t * 4 / period, t)
    value = torch.where(torch.logical_and(t <= period * 3 / 4, t > period / 4), 2 - 4 / period * t, value)
    value = torch.where(torch.logical_and(t <= period, t > period * 3 / 4), 4 / period * t - 4, value)
    return value

def Tan(t, period):
    t = t % period
    value = torch.where(t <= period / 4, torch.tan((t/period)*math.pi), t)
    value = torch.where(torch.logical_and(t <= period * 3 / 4, t > period / 4), -torch.tan((t/period - 1/2)*math.pi), value)
    value = torch.where(torch.logical_and(t <= period, t > period * 3 / 4), torch.tan((t/period - 1)*math.pi), value)
    return value

class DynamicSynapse(Optimizer):

    def __init__(self, params, lr=1e-3, period=None, t_in_period=None, period_var=0.1, amp=1,
                 weight_centre_update_rate=1,
                 weight_oscillate_decay=1e-2, dt=15, oscillating=True, using_range_adapter=False, plot_reward=True,
                 range_adapter_warmup_time=0, alpha_0=-0.5, alpha_1=0.5, a=1, b=-1):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr,
                        period=period,
                        t_in_period=t_in_period,
                        period_var=period_var,
                        amp=amp,
                        weight_centre_update_rate=weight_centre_update_rate,
                        weight_oscillate_decay=weight_oscillate_decay,
                        dt=dt,
                        oscillating=oscillating)
        super(DynamicSynapse, self).__init__(params, defaults)
        self.using_range_adapter = using_range_adapter
        # if self.using_range_adapter:
        #     self.range_adapter = RangeAdapter(targe_max_output=1, targe_min_output=-1, factor=1, bias=0,
        #                                       update_rate=0.1*lr, t=0, name='reward_adapter')
            # if plot_reward:
            #     self.range_adapter.init_recording(log_path='', log_name='reward_adapter.pkl')
        self.modulator_amount_osci = 0
        self.plot_reward = plot_reward
        self.amp = amp
        self.t = 0
        self.dt = dt
        self.range_adapter_warmup_time = range_adapter_warmup_time
        self.alpha_0 = alpha_0
        self.alpha_1 = alpha_1
        self.a = a
        self.b = b

    @torch.no_grad()
    def _init_states(self, p, period=None, t_in_period=None, period_var=None,
                     amp=None, weighter_centre_update_rate=0.000012,
                     weighter_oscillate_decay=0.0000003, oscillating=True):
 
        if period is not None:
            temp_period_centre = torch.mul(torch.ones_like(p, memory_format=torch.preserve_format), period)
        else:
            temp_period_centre = 1000 + 100 * (torch.rand_like(p) - 0.5)
        temp_period = copy.deepcopy(temp_period_centre)
        if t_in_period is not None:
            temp_t_in_period = torch.mul(torch.ones_like(p, memory_format=torch.preserve_format), t_in_period)
        else:
            temp_t_in_period = torch.mul(torch.rand_like(p, memory_format=torch.preserve_format), temp_period)
        if period_var is not None:
            temp_period_var = torch.mul(torch.ones_like(p, memory_format=torch.preserve_format), period_var)
        else:
            temp_period_var = torch.mul(torch.ones_like(p, memory_format=torch.preserve_format), 0.1)
        if amp is not None:
            temp_amp = torch.mul(torch.ones_like(p, memory_format=torch.preserve_format), amp)
        else:
            temp_amp = torch.mul(torch.ones_like(p, memory_format=torch.preserve_format), 0.2)
        temp_weight_centre = copy.deepcopy(p)
        if weighter_centre_update_rate is not None:
            temp_weight_centre_update_rate = torch.mul(torch.ones_like(p, memory_format=torch.preserve_format),
                                                       weighter_centre_update_rate)
        else:
            temp_weight_centre_update_rate = torch.mul(torch.ones_like(p, memory_format=torch.preserve_format),
                                                       0.00001)
        if oscillating:
            temp_weight = temp_weight_centre + temp_amp * torch.sin(temp_t_in_period / temp_period * 2 * math.pi)
        else:
            temp_weight = temp_weight_centre
        if weighter_oscillate_decay is not None:
            temp_weight_oscilate_decay = torch.mul(torch.ones_like(p, memory_format=torch.preserve_format),
                                                   weighter_oscillate_decay)
        else:
            temp_weight_oscilate_decay = torch.mul(torch.ones_like(p, memory_format=torch.preserve_format), 0.0000003)
        temp_zero_cross = torch.ones_like(p, memory_format=torch.preserve_format, dtype=torch.bool)
        temp_time = 0
        return temp_period_centre, temp_period, temp_t_in_period, temp_period_var, temp_amp, temp_weight_centre, \
               temp_weight_centre_update_rate, temp_weight, temp_weight_oscilate_decay, \
               temp_zero_cross, temp_time

    @torch.no_grad()
    def step(self, closure=None, dt=None):

        if dt is None:
            dt = self.dt
        self.t += dt
        reward = None
        if closure is not None:
            reward, alpha  = closure()

        if torch.is_tensor(reward):
            reward = reward.numpy()

        if alpha<self.alpha_0:
            self.mode = 0
        if alpha<=self.alpha_1 and alpha>=self.alpha_0:
            self.mode = 1
        if alpha > self.alpha_1:
            self.mode = 2
        
        if self.using_range_adapter:
            modulator_amount_osci = self.range_adapter.step_dynamics(self.defaults['dt'], reward, factor_rate=0.1)
            if self.plot_reward:
                self.range_adapter.recording()
            self.range_adapter.update()
            if self.range_adapter.t <self.range_adapter_warmup_time:
                return reward
        else:
            modulator_amount_osci = reward
        # if modulator_amount_osci>10:
        #     modulator_amount_osci = 10
        # elif modulator_amount_osci<-10:
        #     modulator_amount_osci = -10
        # modulator_amount_osci = np.tanh(modulator_amount_osci)
        self.modulator_amount_osci = modulator_amount_osci
        if self.t % 100 == 99:
            if self.using_range_adapter:
                print('time', str(self.t), 'loss: ', str(reward), 'modulator_amount_osci: ', str(modulator_amount_osci),
                      'factor', str(self.range_adapter.current_factor),
                      'bias', str(self.range_adapter.current_bias), 'amp', str(self.amp))
            else:
                print('time', str(self.t), 'loss: ', str(reward), 'modulator_amount_osci: ', str(modulator_amount_osci),
                      'amp', str(self.amp))
        for group in self.param_groups:

            if not group['oscillating']:
                continue

            # for p in group['params']:

            #     state = self.state[p]

            #     # State initialization
            #     if len(state) == 0:
            #         state['period_centre'], state['period'], state['t_in_period'], state['period_var'], \
            #         state['amp'], state['weight_centre'], state['weight_centre_update_rate'], \
            #         state['weight'], state['weight_oscilate_decay'], \
            #         state['zero_cross'], state['time'] \
            #             = self._init_states(p, period=group['period'], t_in_period=group['t_in_period'],
            #                                 period_var=group['period_var'],
            #                                 amp=group['amp'],
            #                                 weighter_centre_update_rate=group['weight_centre_update_rate'],
            #                                 weighter_oscillate_decay=group['weight_oscillate_decay'],
            #                                 oscillating=group['oscillating'])

            #     period_centre, period, t_in_period, period_var, \
            #     amp, weight_centre, weight_centre_update_rate, \
            #     weight, weight_oscilate_decay, \
            #     zero_cross, time \
            #         = state['period_centre'], state['period'], state['t_in_period'], state['period_var'], \
            #           state['amp'], state['weight_centre'], state['weight_centre_update_rate'], \
            #           state['weight'], state['weight_oscilate_decay'], \
            #           state['zero_cross'], state['time']

            #     time += group['dt']
            #     t_in_period += group['dt']

            #     # weight = weight_centre *(1+ amp * torch.sin(t_in_period / period * 2 * math.pi))
            #     weight = weight_centre + amp * torch.sin(t_in_period / period * 2 * math.pi)
            #     # weight_centre_var = (weight - weight_centre) \
            #     #                          * modulator_amount_osci * weight_centre_update_rate * group['dt'] * group['lr']
            #     weight_centre_var = (weight - weight_centre) \
            #                         * torch.tanh(modulator_amount_osci * weight_centre_update_rate * group['dt'] * group['lr'])
            #     # if debug:
            #     #     assert not torch.any(torch.isnan(weight_centre)), "weighter_centre has nan" + \
            #     #         str(torch.any(torch.isnan(weight_centre))) + "weight_centre_var=" + \
            #     #         str(weight_centre_var) + "\nweighter_centre" + str(weight_centre)
            #     #     assert not torch.any(torch.isnan(weight_centre_var)), "weight_centre_var has nan" + \
            #     #         "weight_centre_var=" + str(weight_centre_var) + \
            #     #          "\nweighter_centre" + str(weight_centre)
            #     weight_centre += weight_centre_var
            #     amp *= torch.exp(-weight_oscilate_decay * modulator_amount_osci * group['dt'] * group['lr'])
            #     zero_cross = torch.logical_and(torch.less(p, weight_centre),
            #                                    torch.greater_equal(weight, weight_centre))
            #     if torch.any(zero_cross):
            #         t_in_period[zero_cross] = t_in_period[zero_cross] % period[zero_cross]
            #         period[zero_cross] = torch.normal(mean=period_centre[zero_cross],
            #                                                  std=period_centre[zero_cross] * period_var[zero_cross])
            #     p.data = weight.data
            # self.amp = state['amp']
            for i in range(len(group['params'])):
                p = group['params'][i]

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['period_centre'], state['period'], state['t_in_period'], state['period_var'], \
                    state['amp'], state['weight_centre'], state['weight_centre_update_rate'], \
                    state['weight'], state['weight_oscilate_decay'], \
                    state['zero_cross'], state['time'] \
                        = self._init_states(p, period=group['period'], t_in_period=group['t_in_period'],
                                            period_var=group['period_var'],
                                            amp=group['amp'][i],
                                            weighter_centre_update_rate=group['weight_centre_update_rate'],
                                            weighter_oscillate_decay=group['weight_oscillate_decay'],
                                            oscillating=group['oscillating'])

                period_centre, period, t_in_period, period_var, \
                amp, weight_centre, weight_centre_update_rate, \
                weight, weight_oscilate_decay, \
                zero_cross, time \
                    = state['period_centre'], state['period'], state['t_in_period'], state['period_var'], \
                      state['amp'], state['weight_centre'], state['weight_centre_update_rate'], \
                      state['weight'], state['weight_oscilate_decay'], \
                      state['zero_cross'], state['time']

                time += group['dt']
                t_in_period += group['dt']

                # weight = weight_centre *(1+ amp * torch.sin(t_in_period / period * 2 * math.pi))
                # TODO Change oscillating way
                # weight = weight_centre + amp * torch.sin(t_in_period / period * 2 * math.pi)
                # weight = weight_centre + amp * f(t_in_period, period)
                weight = weight_centre + amp * Tan(t_in_period, period)

                # weight_centre_var = (weight - weight_centre) \
                #                          * modulator_amount_osci * weight_centre_update_rate * group['dt'] * group['lr']
                weight_centre_var = (weight - weight_centre) \
                                    * torch.tanh(modulator_amount_osci * weight_centre_update_rate * group['dt'] * group['lr'])
                # if debug:
                #     assert not torch.any(torch.isnan(weight_centre)), "weighter_centre has nan" + \
                #         str(torch.any(torch.isnan(weight_centre))) + "weight_centre_var=" + \
                #         str(weight_centre_var) + "\nweighter_centre" + str(weight_centre)
                #     assert not torch.any(torch.isnan(weight_centre_var)), "weight_centre_var has nan" + \
                #         "weight_centre_var=" + str(weight_centre_var) + \
                #          "\nweighter_centre" + str(weight_centre)
                weight_centre += weight_centre_var

                beta = -weight_oscilate_decay * modulator_amount_osci * group['lr']

                if self.mode == 0:
                    amp *= torch.exp((self.b + beta) * group['dt'])
                if self.mode == 1:
                    amp *= torch.exp((self.a + beta) * group['dt'])
                if self.mode == 2:
                    amp *= torch.exp(beta * group["dt"])
                if self.t % (1000 * dt) == self.dt * 0 and i == (len(group['params']) - 1):
                    self.a *= 0.9698
                    print('\n' + "=="*25 + " 1000 step " + "=="*25)
                    # print(t_in_period[0])
                    print("a:%.11f, b:%.11f, beta:%.11f, a + beta:%.11f, b + beta:%.11f" 
                          %(self.a, self.b, beta.cpu().detach().numpy()[-1], self.a+beta.cpu().detach().numpy()[-1], self.b+beta.cpu().detach().numpy()[-1]))
                # amp *= torch.exp(-weight_oscilate_decay * modulator_amount_osci * group['dt'] * group['lr'])
                zero_cross = torch.logical_and(torch.less(p, weight_centre),
                                               torch.greater_equal(weight, weight_centre))
                if torch.any(zero_cross):
                    t_in_period[zero_cross] = t_in_period[zero_cross] % period[zero_cross]
                    period[zero_cross] = torch.normal(mean=period_centre[zero_cross],
                                                             std=period_centre[zero_cross] * period_var[zero_cross])
                p.data = weight.data
            self.amp = state['amp']
        return reward


# class RangeAdapter:
#
#     def __init__(self, targe_max_output=0.5, targe_min_output=-0.5, factor=1, bias=0,
#                  update_rate=0.0005, t=0, name='RangeAdapter'):
#         self.targe_max_output = targe_max_output
#         self.targe_min_output = targe_min_output
#         self.targe_output_bias = (self.targe_max_output + self.targe_min_output) / 2
#         self.targe_output_range = (self.targe_max_output - self.targe_min_output)
#         self.current_factor = torch.tensor(factor, dtype=torch.float32)
#         self.current_bias = torch.tensor(bias, dtype=torch.float32)
#         self.last_factor = torch.tensor(factor, dtype=torch.float32)
#         self.last_bias = torch.tensor(bias, dtype=torch.float32)
#         self.current_output = None
#         self.last_output = self.current_output
#         self.current_input = None
#         self.last_input = self.current_input
#         self.t = t
#         self.update_rate = update_rate
#         self.name = name
#
#     @torch.no_grad()
#     def step_dynamics(self, dt, input_value):
#         self.t += dt
#         self.current_input = input_value
#         # factor1 = (torch.log10(self.current_factor) - (-2)) * (2 - torch.log10(self.current_factor)) / 4
#         factor1= 1
#         self.current_bias += (self.current_input - self.last_bias) * self.update_rate * dt * factor1*100
#         self.current_factor += (self.targe_output_range * 0.5 - self.last_factor \
#                                 * torch.abs(self.current_input - self.current_bias)) \
#                                * dt * self.update_rate * factor1
#         self.current_factor[self.current_factor < torch.finfo(self.current_factor.dtype).tiny] \
#             = torch.finfo(self.current_factor.dtype).tiny
#         self.current_output = (input_value - self.last_bias) * self.current_factor
#         return self.current_output
#
#     @torch.no_grad()
#     def update(self):
#         self.last_factor = self.current_factor
#         self.last_bias = self.current_bias
#         self.last_input = self.current_input
#         self.last_output = self.current_output
