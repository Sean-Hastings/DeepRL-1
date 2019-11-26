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

    def geoff_pac_update(self, s, a, mu_a, r, next_s, m):
        # V and C networks are bundled together, so I'm gonna update both together by just summing the losses described in the paper (pseudocode)

        ### Inputs from config object
        gamma = self.config.discount
        gamma_hat = self.config.gamma_hat
        beta  = self.config.c_coef

        ### Setup for update steps
        # Detached because it's actually from policy network, separate from both C and V networks
        rho_t = self.network(s, a)['pi_a'].detach() / mu_a
        # Detach target so we don't update it
        delta_t = r + gamma * self.target_network(next_s)['v'].detach() - self.network(s)['v']
        sn_loss = ? # https://arxiv.org/pdf/1901.09455.pdf ; section: Soft Ratio Normalization ; Equation 7?

        ### Network update losses
        # Note: I don't like that rho_t is in both of these. I think it should be detached in one of them to avoid cross-contamination
        v_loss = rho_t * delta_t * delta_t
        c_loss = ((gamma_hat * rho_t * self.target_network(s)['c'].detach() + (1 - gamma_hat) - self.network(next_s)['c'])**2 + beta * sn_loss)
        full_network_loss = v_loss + c_loss

        ### backprop + parameter update
        self.optimizer.zero_grad()
        full_network_loss.backward()
        self.optimizer.step()

        # Note: rho_t was detached because it wasn't from the parameters being updated, but the parameters updated after this point do lead to it, so should it be recalculated and not detached??

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
