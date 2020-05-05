import os
import fire
import numpy as np
import logging
import gym
import pandas as pd
import pyarrow.parquet as pq  # noqa: F401
from azureml.studio.core.io.data_frame_directory import save_data_frame_to_directory
import torch
from ddpg.model import Actor

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s',
                    datefmt='%d-%M-%Y %H:%M:%S', level=logging.INFO)


class Score:
    def __init__(self, model_path, meta={}):
        self.env = gym.make(meta['Environment name'])
        self.gamma = float(meta['Gamma'])
        self.validation_times = int(meta['Validation times'])
        s_dim = self.env.observation_space.shape[0]
        a_dim = self.env.action_space.shape[0]
        a_max = self.env.action_space.high[0]
        self.actor = Actor(s_dim, a_dim, a_max)
        self.actor.load_state_dict(torch.load(os.path.join(model_path, 'actor.pth'), map_location='cpu'))
        self.actor.eval()

    def run(self, meta=None):
        my_list = [[self.validation_times, self.gamma]]
        max_steps = 5000
        total_temp_reward = 0
        for i in range(self.validation_times):
            temp_reward = 0
            observation = self.env.reset()
            for r in range(max_steps):
                state = np.float32(observation)
                with torch.no_grad():
                    state = torch.from_numpy(state)
                    action = self.actor(state).detach()
                    action = action.data.numpy()
                new_observation, reward, done, info = self.env.step(action)
                observation = new_observation
                temp_reward = temp_reward * self.gamma + reward
                if done:
                    break
            total_temp_reward += temp_reward
        total_temp_reward /= self.validation_times
        my_list[0].append(total_temp_reward)
        logging.info('Validation times: {}, Gamma: {}, Mean reward: {}'.format(self.validation_times,
                                                                               self.gamma, total_temp_reward))
        df = pd.DataFrame(my_list, columns=['Validation times', 'Gamma', 'Mean reward'])
        return df

    def inference(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        df = self.run()
        save_data_frame_to_directory(save_path, data=df)


def test(environment_name='Pendulum-v0', validation_times=10, gamma=0.99,
         model_path='ddpg/saved_path', save_path='ddpg/outputs'):
    meta = {'Environment name': str(environment_name), 'Validation times': str(validation_times), 'Gamma': str(gamma)}
    score = Score(model_path, meta)
    score.inference(save_path=save_path)
    logging.info('This experiment has been completed.')


if __name__ == '__main__':
    fire.Fire(test)
