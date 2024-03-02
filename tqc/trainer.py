import torch
import time
from tqc.functions import quantile_huber_loss_f
from tqc import DEVICE
from tqc.dynamicsynapse import DynamicSynapse
import dill
# nowtime = time.strftime("%m-%d_%H-%M-%S", time.localtime())

def closure(r):
    def a():
        # out = adapter.step_dynamics(dt, r)
        # adapter.update()
        return r
    return a
class Trainer(object):
	def __init__(
		self,
		*,
		actor,
		critic,
		critic_target,
		critic_value,
		critic_value_target,
		discount,
		tau,
		top_quantiles_to_drop,
		target_entropy,
		trace_path
		# replay_buffer
	):
		self.actor = actor
		self.critic = critic
		self.critic_target = critic_target
		self.critic_value = critic_value
		self.critic_value_target = critic_value_target
		self.log_alpha = torch.zeros((1,), requires_grad=True, device=DEVICE)
		self.trace_path = trace_path
		# self.replay_buffer = replay_buffer

		# TODO: check hyperparams
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
		self.critic_value_optimizer = torch.optim.Adam(self.critic_value.parameters(), lr=3e-4)
		self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)
		self.dynamic_optimizer = DynamicSynapse(self.actor.parameters(), dt=15)

		self.discount = discount
		self.tau = tau
		self.top_quantiles_to_drop = top_quantiles_to_drop
		self.target_entropy = target_entropy

		self.quantiles_total = critic.n_quantiles * critic.n_nets

		self.total_it = 0

	def train(self, replay_buffer, batch_size=256, max_step=1e6):
		# start = time.time()
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
		alpha = torch.exp(self.log_alpha)
		# end = time.time()
		# print(end - start)

		# --- Q loss ---
		with torch.no_grad():
			# get policy action
			new_next_action, next_log_pi = self.actor(next_state)

			# compute and cut quantiles at the next state
			next_z_value = self.critic_value_target(next_state)
			next_z = self.critic_target(next_state, new_next_action)
			sorted_z_value, _ = torch.sort(next_z_value.reshape(batch_size, -1))
			# batch x nets x quantiles
			sorted_z, _ = torch.sort(next_z.reshape(batch_size, -1))
			sorted_z_part = sorted_z[:, :self.quantiles_total-self.top_quantiles_to_drop]
			sorted_z_value_part = sorted_z_value[:, :self.quantiles_total-self.top_quantiles_to_drop]
			# print(sorted_z_part)
			# compute target
			target = reward + not_done * self.discount * (sorted_z_part - alpha * next_log_pi)
			target_value = reward + not_done * self.discount * (sorted_z_value_part - alpha * next_log_pi)

		cur_z = self.critic(state, action)
		# print("cur_z:  ", cur_z)
		cur_z_value = self.critic_value(state)
		# print("cur_z:{}  cur_z_value:{}".format(cur_z.mean().cpu().detach().item(), cur_z_value.mean().cpu().detach().item()))
		critic_loss = quantile_huber_loss_f(cur_z, target)
		critic_value_loss = quantile_huber_loss_f(cur_z_value, target_value)
		# print("lossL  ", critic_loss)

		# --- Policy and alpha loss ---
		new_action, log_pi = self.actor(state)
		alpha_loss = -self.log_alpha * (log_pi + self.target_entropy).detach().mean()
		actor_loss = (alpha * log_pi - self.critic(state, new_action).mean(2).mean(1, keepdim=True)).mean()

		# --- Update ---
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		self.critic_value_optimizer.zero_grad()
		critic_value_loss.backward()
		self.critic_value_optimizer.step()

		for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		for param, target_param in zip(self.critic_value.parameters(), self.critic_value_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		# TODO
		if self.total_it >= max_step - batch_size - 600 :
			for name, param in self.actor.named_parameters():
				if "weight" in name or "bias" in name:
					self.actor.Trace["name"].append(name)
					self.actor.Trace["weight"].append(param.data)
					self.actor.Trace["gradient"].append(param.grad)

		if self.total_it == max_step - batch_size - 1:
			# self.trace_path = ChooseTracePath() + "trace_"+self.env_name+"_"+str(self.total_step)+".pkl"
			with open(self.trace_path, 'wb') as f:
				dill.dump(self.actor.Trace, f) 
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		self.alpha_optimizer.zero_grad()
		alpha_loss.backward()
		self.alpha_optimizer.step()

		self.total_it += 1
		# print(self.total_it)
	
	def continue_train(self, replay_buffer, batch_size, trace):
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
		alpha = torch.exp(self.log_alpha)

		# --- Q loss ---
		with torch.no_grad():
			# get policy action
			new_next_action, next_log_pi = self.actor(state)

			# compute and cut quantiles at the next state
			next_z_value = self.critic_value_target(next_state)
			next_z = self.critic_target(next_state, new_next_action)
			sorted_z_value, _ = torch.sort(next_z_value.reshape(batch_size, -1))
			# batch x nets x quantiles
			sorted_z, _ = torch.sort(next_z.reshape(batch_size, -1))
			sorted_z_part = sorted_z[:, :self.quantiles_total-self.top_quantiles_to_drop]
			sorted_z_value_part = sorted_z_value[:, :self.quantiles_total-self.top_quantiles_to_drop]
			# print(sorted_z_part)
			# compute target
			target = reward + not_done * self.discount * (sorted_z_part - alpha * next_log_pi)
			target_value = reward + not_done * self.discount * (sorted_z_value_part - alpha * next_log_pi)

		cur_z = self.critic(state, action)
		# print("cur_z:  ", cur_z)
		cur_z_value = self.critic_value(state)

		critic_loss = quantile_huber_loss_f(cur_z, target)
		critic_value_loss = quantile_huber_loss_f(cur_z_value, target_value)
		# print("lossL  ", critic_loss)

		# --- Policy and alpha loss ---
		new_action, log_pi = self.actor(state)
		alpha_loss = -self.log_alpha * (log_pi + self.target_entropy).detach().mean()
		# actor_loss = (alpha * log_pi - self.critic(state, new_action).mean(2).mean(1, keepdim=True)).mean()
		actor_loss = (cur_z - cur_z_value).reshape(batch_size, -1).mean().cpu().detach()
		print("cur_z:{}  cur_z_value:{}".format(cur_z.mean().cpu().detach().item(), cur_z_value.mean().cpu().detach().item()))
		# actor_loss *= 1e-5
		# print(actor_loss)
		trace["Q"].append(cur_z.mean().cpu().detach().item())
		trace["advantage"].append(actor_loss)
		# --- Update ---
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		self.critic_value_optimizer.zero_grad()
		critic_value_loss.backward()
		self.critic_value_optimizer.step()


		for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		for param, target_param in zip(self.critic_value.parameters(), self.critic_value_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		# self.actor_optimizer.zero_grad()
		# actor_loss.backward()
		# self.actor_optimizer.step()
		self.dynamic_optimizer.step(closure(actor_loss))

		self.alpha_optimizer.zero_grad()
		alpha_loss.backward()
		self.alpha_optimizer.step()

		self.total_it += 1

	def save(self, filename, nowtime):
		filename = str(filename)
		torch.save(self.critic.state_dict(), filename + "_critic" + nowtime)
		torch.save(self.critic_target.state_dict(), filename + "_critic_target"+ nowtime)
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer"+ nowtime)
		torch.save(self.critic_value.state_dict(), filename + "_critic_value"+ nowtime)
		torch.save(self.critic_value_target.state_dict(), filename + "_critic_value_target"+ nowtime)
		torch.save(self.critic_value_optimizer.state_dict(), filename + "_critic_value_optimizer"+ nowtime)
		torch.save(self.actor.state_dict(), filename + "_actor"+ nowtime)
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer"+ nowtime)
		torch.save(self.log_alpha, filename + '_log_alpha'+ nowtime)
		torch.save(self.alpha_optimizer.state_dict(), filename + "_alpha_optimizer"+ nowtime)
		

	def load(self, filename, logtime):
		filename = str(filename)
		self.critic.load_state_dict(torch.load(filename + "_critic"+logtime))
		self.critic_target.load_state_dict(torch.load(filename + "_critic_target"+logtime))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"+logtime))
		self.actor.load_state_dict(torch.load(filename + "_actor"+logtime))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"+logtime))
		self.log_alpha = torch.load(filename + '_log_alpha'+logtime)
		self.alpha_optimizer.load_state_dict(torch.load(filename + "_alpha_optimizer"+logtime))
		self.critic_value.load_state_dict(torch.load(filename + "_critic_value"+logtime))
		self.critic_value_target.load_state_dict(torch.load(filename + "_critic_value_target"+logtime))
		self.critic_value_optimizer.load_state_dict(torch.load(filename + "_critic_value_optimizer"+logtime))	
