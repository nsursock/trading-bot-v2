from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import logging
import traceback
from callbacks import LogMetricsCallback, EarlyStoppingCallback
import math
import time
import os
from reporting import plot_training_metrics

import cProfile
import pstats

class TradingAgent:
    def __init__(self, train_env, eval_env, test_env, training_params=None, financial_params=None, output_dir='.'):
        # Wrap the environment with Monitor and DummyVecEnv
        self.train_env = DummyVecEnv([lambda: Monitor(train_env)])
        self.train_env = VecNormalize(self.train_env, norm_obs=True, norm_reward=True, clip_obs=10.)
        
        # Extract initial values for clip_range and learning_rate
        initial_clip_range = training_params.get('clip_range', 0.2)  # Default to 0.2 if not provided
        initial_learning_rate = training_params.get('learning_rate', 3e-4)  # Default to 3e-4 if not provided

        # Create schedule functions with initial values
        clip_range_schedule = self.get_schedule_function(training_params.get('clip_range_schedule', 'exponential'))(initial_clip_range)
        learning_rate_schedule = self.get_schedule_function(training_params.get('learning_rate_schedule', 'linear'))(initial_learning_rate)

        # Remove 'clip_range' and 'learning_rate' from training_params if they exist
        hyper_params = {
            'n_steps': training_params.get('n_steps'),
            'batch_size': training_params.get('batch_size'),
            'n_epochs': training_params.get('n_epochs'),
            'gae_lambda': training_params.get('gae_lambda'),
            'ent_coef': training_params.get('ent_coef'),
            'vf_coef': training_params.get('vf_coef'),
            'max_grad_norm': training_params.get('max_grad_norm')
        }

        if financial_params:
            selected_params = financial_params.copy()

        mode = training_params.get('training_mode', 'standard')
        if training_params['train_model']:
            if mode == 'custom':
                self.model = PPO('MlpPolicy', self.train_env, **hyper_params, clip_range=clip_range_schedule, learning_rate=learning_rate_schedule, verbose=1)
            else:
                self.model = PPO('MlpPolicy', self.train_env, clip_range=clip_range_schedule, learning_rate=learning_rate_schedule, verbose=1)
        else:
            self.model = PPO.load(os.path.join(output_dir, selected_params['model_name']))
        
        self.eval_env = DummyVecEnv([lambda: Monitor(eval_env)])
        self.eval_env = VecNormalize(self.eval_env, norm_obs=True, norm_reward=True, clip_obs=10.)
        
        self.test_env = DummyVecEnv([lambda: Monitor(test_env)])
        self.test_env = VecNormalize(self.test_env, norm_obs=True, norm_reward=True, clip_obs=10.)
        
    @staticmethod
    def linear_schedule(initial_value=3e-4):
        """
        Linear learning rate schedule.
        :param initial_value: (float) initial learning rate
        :return: (function)
        """
        def func(progress_remaining):
            # progress_remaining goes from 1 to 0
            return progress_remaining * initial_value
        return func

    @staticmethod
    def exponential_schedule(initial_value=3e-4, decay_rate=0.01):
        """
        Exponential learning rate schedule.
        :param initial_value: (float) initial learning rate
        :param decay_rate: (float) decay rate for each step
        :return: (function)
        """
        def func(progress_remaining):
            return initial_value * math.exp(-decay_rate * (1 - progress_remaining))
        return func
    
    @staticmethod
    def step_schedule(initial_value=3e-4, drop_rate=0.5, drop_every=0.2):
        """
        Step learning rate schedule.
        :param initial_value: (float) initial learning rate
        :param drop_rate: (float) factor by which to drop the learning rate
        :param drop_every: (float) fraction of progress when to drop the rate
        :return: (function)
        """
        def func(progress_remaining):
            factor = int(progress_remaining / drop_every)
            return initial_value * (drop_rate ** factor)
        return func

    def get_schedule_function(self, schedule_type):
        """
        Returns the appropriate schedule function based on the schedule type.
        :param schedule_type: (str) The type of schedule ('linear', 'exponential', 'step')
        :return: (function) The schedule function
        """
        if schedule_type == 'linear':
            return self.linear_schedule
        elif schedule_type == 'exponential':
            return self.exponential_schedule
        elif schedule_type == 'step':
            return self.step_schedule
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
    def profile_train(self, timesteps=10000, output_dir="."):
        profiler = cProfile.Profile()
        profiler.enable()
        
        # Call the method you want to profile
        result = self.train(timesteps, output_dir)
        
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats(20)  # Print the top 10 functions by cumulative time
        
        return result

    def train(self, timesteps=10000, output_dir="."):
        training_metrics = {
            'approx_kl': [],
            'clip_fraction': [],
            'clip_range': [],
            'entropy_loss': [],
            'explained_variance': [],
            'learning_rate': [],
            'loss': [],
            'n_updates': [],
            'policy_gradient_loss': [],
            'value_loss': []
        }
        
        log_metrics_callback = LogMetricsCallback(training_metrics, timesteps)
        early_stopping_callback = EarlyStoppingCallback(self.eval_env, eval_freq=10000, patience=5)
        self.model.learn(total_timesteps=timesteps, callback=[log_metrics_callback, early_stopping_callback])
        self.model.save(os.path.join(output_dir, "model_ppo_crypto_trading"))
        plot_training_metrics(training_metrics, save_path=os.path.join(output_dir, "training_metrics.png"))

    def profile_evaluate(self, episodes=10):
        profiler = cProfile.Profile()
        profiler.enable()
        
        # Call the method you want to profile
        result = self.evaluate(episodes)
        
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats(20)  # Print the top 10 functions by cumulative time
        
        return result

    def evaluate(self, episodes=10):
        actions_history = []
        rewards_history = []
        episode_durations = []  # List to store the duration of each episode
        balances = []
        net_worths = []
        try:
            logging.info(f"Starting evaluation of {episodes} episodes")
            for episode in range(episodes):
                start_time = time.time()  # Record the start time of the episode
                obs = self.test_env.reset()
                done = False
                
                while not done:
                    # Predict an action for each symbol
                    actions, _states = self.model.predict(obs)
                    obs, reward, done, _, info = self.test_env.envs[0].step(actions)
                    self.test_env.render()
                    
                    # Collect balance and net worth from info
                    balances.append(info.get('balance', 0))
                    net_worths.append(info.get('net_worth', 0))
                    
                    actions_history.append(actions)
                    rewards_history.append(reward)
                
                end_time = time.time()  # Record the end time of the episode
                episode_duration = end_time - start_time  # Calculate the duration
                episode_durations.append(episode_duration)  # Store the duration

                # Calculate average duration and ETA
                avg_duration = sum(episode_durations) / len(episode_durations)
                remaining_episodes = episodes - (episode + 1)
                eta = avg_duration * remaining_episodes

                logging.info(f"Finished evaluation of episode {episode + 1} out of {episodes} in {episode_duration:.2f} seconds. ETA: {eta:.2f} seconds")
        except Exception as e:
            logging.error(f"Evaluation failed with exception: {e}")
            logging.error("Traceback details:")
            logging.error(traceback.format_exc())
            raise
        
        return actions_history, rewards_history, episode_durations, balances, net_worths