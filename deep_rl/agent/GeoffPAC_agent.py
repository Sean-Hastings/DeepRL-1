#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *


class GeoffPACAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.total_steps = 0
        self.states = self.task.reset()

        self.episode_rewards = []
        self.online_rewards = np.zeros(config.num_workers)

        self.replay = config.replay_fn()

        self.F1 = torch.zeros((config.num_workers, 1), device=Config.DEVICE)
        self.F2 = Grads(self.network, config.num_workers)
        self.grad_prev = Grads(self.network, config.num_workers)
        self.rho_prev = torch.zeros(self.F1.size(), device=Config.DEVICE)
        self.c_prev = torch.zeros(self.F1.size(), device=Config.DEVICE)

    def random_action(self):
        config = self.config
        action = [config.eval_env.action_space.sample() for _ in range(config.num_workers)]
        return np.asarray(action)

    def learn_v_c_batch(self, transitions):
        for i in range(self.config.vc_epochs):
            self.learn_v_c(self.replay.sample())
        self.learn_v_c(transitions)

    def learn_v_c(self, transitions):
        config = self.config
        s, a, mu_a, r, next_s, m = transitions

        prediction = self.network(s, a)
        next_prediction = self.network(next_s)
        prediction_target = self.target_network(s, a)
        next_prediction_target = self.target_network(next_s)

        rho = prediction['pi_a'] / mu_a
        rho = rho.detach()

        v_target = next_prediction_target['v']
        v_target = r + config.discount * m * v_target
        v_target = v_target.detach()
        td_error = v_target - prediction['v']
        # config.logger.add_histogram('v', prediction['v'])
        # config.logger.add_histogram('c', prediction['c'])
        # config.logger.add_histogram('rho', rho)
        v_loss = td_error.pow(2).mul(0.5).mul(rho.clamp(0, 1)).mean()

        c_target = config.gamma_hat * rho.clamp(0, 2) * prediction_target['c'] + 1 - config.gamma_hat
        c_target = c_target.detach()
        c_next = next_prediction['c'] * m

        c_normalizer = (c_next.sum(-1).unsqueeze(-1) - c_next).mul(1 / (c_next.size(0) - 1)) - 1
        c_normalizer = c_normalizer.detach() * c_next

        c_loss = (c_target - c_next).pow(2).mul(0.5).mean() + config.c_coef * c_normalizer.mean()

        self.optimizer.zero_grad()
        (v_loss + c_loss).backward()
        self.optimizer.step()

    def off_pac_update(self, s, a, mu_a, r, next_s, m):
        config = self.config
        self.learn_v_c_batch([s, a, mu_a, r, next_s, m])

        prediction = self.network(s, a)
        with torch.no_grad():
            target = self.target_network(next_s)['v']
            target = r + config.discount * m * target
            rho = prediction['pi_a'] / mu_a
            rho = rho.clamp(0, 2)
        td_error = target - prediction['v']
        entropy = prediction['ent'].mean()
        pi_loss = -rho * td_error.detach() * prediction['log_pi_a'] - config.entropy_weight * entropy
        pi_loss = pi_loss.mean()

        self.optimizer.zero_grad()
        pi_loss.backward()
        self.optimizer.step()

    def ace_update(self, s, a, mu_a, r, next_s, m):
        config = self.config
        self.learn_v_c_batch([s, a, mu_a, r, next_s, m])

        prediction = self.network(s, a)
        with torch.no_grad():
            target = self.target_network(next_s)['v']
            target = r + config.discount * m * target
            self.F1 = m * config.discount * self.rho_prev * self.F1 + 1
            M = (1 - config.lam1) + config.lam1 * self.F1
            rho = prediction['pi_a'] / mu_a
            rho = rho.clamp(0, 2)

        td_error = target - prediction['v']

        entropy = prediction['ent'].mean()
        pi_loss = -M * rho * td_error.detach() * prediction['log_pi_a'] - config.entropy_weight * entropy
        pi_loss = pi_loss.mean()

        self.rho_prev = rho

        self.optimizer.zero_grad()
        pi_loss.backward()
        self.optimizer.step()

    def geoff_pac_update(self, s, a, mu_a, r, next_s, m):
        pass

    def eval_step(self, state):
        with torch.no_grad():
            action = self.network(state)['a']
        if self.config.action_type == 'discrete':
            return to_np(action)
        elif self.config.action_type == 'continuous':
            return to_np(action)
        else:
            raise NotImplementedError

    def step(self):
        config = self.config
        actions = self.random_action()
        if config.action_type == 'discrete':
            mu_a = np.zeros_like(actions) + 1 / config.action_dim
        elif config.action_type == 'continuous':
            mu_a = np.zeros((config.num_workers, )) + 1 / np.power(2.0, config.action_dim)
        else:
            raise NotImplementedError
        next_states, rewards, terminals, _ = self.task.step(actions)
        self.online_rewards += rewards
        rewards = config.reward_normalizer(rewards)
        for i, terminal in enumerate(terminals):
            if terminals[i]:
                self.episode_rewards.append(self.online_rewards[i])
                self.online_rewards[i] = 0

        mask = (1 - terminals).astype(np.uint8)
        transition = [tensor(self.states),
                      tensor(actions),
                      tensor(mu_a).unsqueeze(1),
                      tensor(rewards).unsqueeze(1),
                      tensor(next_states),
                      tensor(mask).unsqueeze(1)]
        if self.replay.size() >= config.replay_warm_up:
            if config.algo == 'off-pac':
                self.off_pac_update(*transition)
            elif config.algo == 'ace':
                self.ace_update(*transition)
            elif config.algo == 'geoff-pac':
                self.geoff_pac_update(*transition)
            else:
                raise NotImplementedError
        self.replay.feed(transition)
        self.states = next_states

        self.total_steps += config.num_workers

        if self.total_steps // config.num_workers % config.target_network_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())
