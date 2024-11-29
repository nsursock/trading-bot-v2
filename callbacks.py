from stable_baselines3.common.callbacks import BaseCallback
import logging
import time
import numpy as np

class EarlyStoppingCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=10000, patience=5, verbose=1):
        super(EarlyStoppingCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.patience = patience
        self.best_mean_reward = -np.inf
        self.no_improvement_steps = 0

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            mean_reward = self._evaluate_policy()
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.no_improvement_steps = 0
            else:
                self.no_improvement_steps += 1

            if self.no_improvement_steps >= self.patience:
                if self.verbose > 0:
                    print(f"Stopping training early after {self.no_improvement_steps} evaluations with no improvement.")
                return False
        return True

    def _evaluate_policy(self):
        episode_rewards = []
        for _ in range(5):  # Evaluate for 5 episodes
            obs = self.eval_env.reset()
            done = False
            total_reward = 0.0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _ = self.eval_env.step(action)
                total_reward += reward
            episode_rewards.append(total_reward)
        mean_reward = np.mean(episode_rewards)
        return mean_reward

class LogMetricsCallback(BaseCallback):
    def __init__(self, training_metrics, total_timesteps, verbose=0):
        super(LogMetricsCallback, self).__init__(verbose)
        self.training_metrics = training_metrics
        self.total_timesteps = total_timesteps
        self.metrics_to_log = [
            'train/approx_kl', 'train/clip_fraction', 'train/clip_range', 'train/entropy_loss',
            'train/explained_variance', 'train/learning_rate', 'train/loss', 'train/n_updates',
            'train/policy_gradient_loss', 'train/value_loss'
        ]
        self.step_times = []
        self.current_timestep = 0

    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_start(self) -> None:
        self.rollout_start_time = time.perf_counter()

    def _on_rollout_end(self) -> None:
        # Access the metrics from the logger
        for key in self.metrics_to_log:
            # Use the logger to retrieve the latest recorded values
            value = self.logger.name_to_value.get(key, None)
            if value is not None:
                # Remove 'train/' prefix for self.training_metrics
                metric_name = key.replace('train/', '')
                if metric_name not in self.training_metrics:
                    self.training_metrics[metric_name] = []
                # Append the value to the training metrics
                self.training_metrics[metric_name].append(value)
                logging.debug(f"Logging {key}: {value}")
            else:
                logging.debug(f"{key} not found in the logger after rollout.")
                
        # Calculate ETA based on rollout duration
        rollout_duration = time.perf_counter() - self.rollout_start_time
        timesteps_done = self.locals['n_steps']
        avg_step_time = rollout_duration / timesteps_done
        remaining_timesteps = self.total_timesteps - self.num_timesteps
        eta = remaining_timesteps * avg_step_time

        # Check if ETA is less than or equal to zero
        if eta <= 0:
            eta_formatted = "00h00m00s"
        else:
            # Convert ETA to hours, minutes, and seconds
            eta_hours, rem = divmod(eta, 3600)
            eta_minutes, eta_seconds = divmod(rem, 60)
            eta_formatted = f"{int(eta_hours):02}h{int(eta_minutes):02}m{int(eta_seconds):02}s"

        logging.info(f"Rollout duration: {rollout_duration:.6f} seconds, ETA: {eta_formatted}")