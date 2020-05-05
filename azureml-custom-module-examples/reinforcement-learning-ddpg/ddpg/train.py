import fire
import os
import torch.nn.functional as F
import gym
import numpy as np
import torch
import logging
import ddpg.utils as utils
import ddpg.model as model
import ddpg.buffer as buffer

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s',
                    datefmt='%d-%M-%Y %H:%M:%S', level=logging.INFO)


class Trainer:

    def __init__(self, state_dim, action_dim, action_lim, ram, learning_rate, gamma):
        """
        :param state_dim: Dimensions of state (int)
        :param action_dim: Dimension of action (int)
        :param action_lim: Used to limit action in [-action_lim,action_lim]
        :param ram: replay memory buffer object
        :return:
        """
        self.TAU = 0.001
        self.GAMMA = gamma
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_lim = action_lim
        self.ram = ram
        self.iter = 0
        self.noise = utils.OrnsteinUhlenbeckActionNoise(self.action_dim)
        self.actor = model.Actor(self.state_dim, self.action_dim, self.action_lim)
        self.target_actor = model.Actor(self.state_dim, self.action_dim, self.action_lim)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), learning_rate)
        self.critic = model.Critic(self.state_dim, self.action_dim)
        self.target_critic = model.Critic(self.state_dim, self.action_dim)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), learning_rate)
        utils.hard_update(self.target_actor, self.actor)
        utils.hard_update(self.target_critic, self.critic)

    def get_exploitation_action(self, state):
        """
        gets the action from target actor added with exploration noise
        :param state: state (Numpy array)
        :return: sampled action (Numpy array)
        """
        state = torch.from_numpy(state)
        action = self.target_actor(state).detach()
        return action.data.numpy()

    def get_exploration_action(self, state):
        """
        gets the action from actor added with exploration noise
        :param state: state (Numpy array)
        :return: sampled action (Numpy array)
        """
        state = torch.from_numpy(state)
        action = self.actor(state).detach()
        new_action = action.data.numpy() + (self.noise.sample() * self.action_lim)
        return new_action

    def optimize(self, batch_size=128):
        """
        Samples a random batch from replay memory and performs optimization
        :return:
        """
        s1, a1, r1, s2 = self.ram.sample(batch_size)
        s1 = torch.from_numpy(s1)
        a1 = torch.from_numpy(a1)
        r1 = torch.from_numpy(r1)
        s2 = torch.from_numpy(s2)
        # ---------------------- optimize critic ----------------------
        # Use target actor exploitation policy here for loss evaluation
        a2 = self.target_actor(s2).detach()
        next_val = self.target_critic(s2, a2).detach()
        r1 = torch.unsqueeze(r1, 1)
        y_expected = r1 + self.GAMMA * next_val
        y_predicted = self.critic(s1, a1)
        # compute critic loss, and update the critic
        loss_critic = F.smooth_l1_loss(y_predicted, y_expected)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()
        # ---------------------- optimize actor ----------------------
        pred_a1 = self.actor(s1)
        loss_actor = -1 * torch.sum(self.critic(s1, pred_a1))
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()
        utils.soft_update(self.target_actor, self.actor, self.TAU)
        utils.soft_update(self.target_critic, self.critic, self.TAU)
        logging.info('Iteration :- ' + str(self.iter) + ' Loss_actor :- ' + str(loss_actor.detach().numpy())
                     + ' Loss_critic :- ' + str(loss_critic.detach().numpy()))
        self.iter += 1

    def save_models(self, save_path='ddpg/saved_model'):
        """
        saves the target actor and critic models
        :return:
        """
        torch.save(self.target_actor.state_dict(), os.path.join(save_path, 'actor.pth'))
        torch.save(self.target_critic.state_dict(), os.path.join(save_path, 'critic.pth'))
        logging.info('Models saved successfully')

    def load_models(self, load_path='ddpg/saved_model'):
        """
        loads the target actor and critic models, and copies them onto actor and critic models
        :return:
        """
        self.actor.load_state_dict(torch.load(os.path.join(load_path, 'actor.pth')))
        self.critic.load_state_dict(torch.load(os.path.join(load_path, 'critic.pth')))
        utils.hard_update(self.target_actor, self.actor)
        utils.hard_update(self.target_critic, self.critic)
        logging.info('Models loaded successfully')

    def train(self):
        self.actor.train()
        self.critic.train()

    def eval(self):
        self.actor.eval()
        self.critic.eval()


def train(environment_name='Pendulum-v0', save_path='ddpg/saved_model', batch_size=128,
          learning_rate=0.001, max_episodes=5000, gamma=0.99):
    env = gym.make(environment_name)
    # env = gym.make('BipedalWalker-v2')
    max_steps = 1000
    max_buffer = 1000000
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_max = env.action_space.high[0]
    print(' State Dimensions :- ', s_dim)
    print(' Action Dimensions :- ', a_dim)
    print(' Action Max :- ', a_max)
    ram = buffer.MemoryBuffer(max_buffer)
    trainer = Trainer(s_dim, a_dim, a_max, ram, learning_rate, gamma)
    best_reward = 0
    has_saved_model = False
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, 'log.csv'), 'w') as f:
        f.write('Episode,Reward\n')
    for _ep in range(max_episodes):
        trainer.train()
        observation = env.reset()
        print('EPISODE :- ', _ep)
        for r in range(max_steps):
            state = np.float32(observation)
            action = trainer.get_exploration_action(state)
            new_observation, reward, done, info = env.step(action)
            if done:
                new_state = None
            else:
                new_state = np.float32(new_observation)
                # push this exp in ram
                ram.add(state, action, reward, new_state)
            observation = new_observation
            # perform optimization
            trainer.optimize(batch_size)
            if done:
                break
        if _ep % 10 == 0:
            trainer.eval()
            val_episodes = 10
            total_temp_reward = 0
            for i in range(val_episodes):
                temp_reward = 0
                observation = env.reset()
                for r in range(max_steps):
                    state = np.float32(observation)
                    with torch.no_grad():
                        action = trainer.get_exploitation_action(state)
                    new_observation, reward, done, info = env.step(action)
                    observation = new_observation
                    temp_reward = temp_reward * gamma + reward
                    if done:
                        break
                total_temp_reward += temp_reward
            total_temp_reward /= val_episodes
            logging.info('Validation reward: ' + str(total_temp_reward))
            with open(os.path.join(save_path, 'log.csv'), 'a') as f:
                f.write('{},{:.4f}\n'.format(_ep, total_temp_reward))
            if not has_saved_model:
                has_saved_model = True
                best_reward = total_temp_reward
                trainer.save_models(save_path)
            else:
                if total_temp_reward > best_reward:
                    trainer.save_models(save_path)
                    best_reward = total_temp_reward
    logging.info('This experiment has been completed.')


if __name__ == '__main__':
    fire.Fire(train)
