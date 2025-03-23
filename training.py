import os
import warnings
from warnings import filterwarnings
import gymnasium as gym

# suppress all warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# suppress noisy logs
os.environ['RAY_DEDUP_LOGS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress TF logs

# specifically for deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# for ray-specific deprecation warnings
os.environ["RAY_DISABLE_DEPRECATION_WARNINGS"] = "1"

# pandas
os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"

# rest of imports
import ray
import torch
import pygame
import numpy as np
import matplotlib.pyplot as plt

import sys
import time
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

# RLlib / Model
from ray.rllib.models import ModelCatalog
from lunabot.geosearch import EnhancedGeosearchModel, GeosearchEnv, NormalizeObservation
# The above import assumes you put both GeosearchEnv + NormalizeObservation in lunabot/geosearch.py
# Adjust your import path accordingly if they're in separate files.

ModelCatalog.register_custom_model("geo_model", EnhancedGeosearchModel)

class RLlibPolicyWrapper:
    """Wrapper for RLlib policies"""
    def __init__(self, algo):
        self.algo = algo
        # Check for CUDA -> MPS -> CPU in priority order
        self.device = "cuda" if torch.cuda.is_available() else \
                    "mps" if torch.backends.mps.is_available() else "cpu"
        
        # Optional: Set MPS-specific flags
        if self.device == "mps":
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            torch.mps.empty_cache()
            
    def get_action(self, state):
        # Convert numpy arrays to tensors on the correct device
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).to(self.device)
            
        return self.algo.compute_single_action(
            observation=state,
            explore=False
        )

import numpy as np
import matplotlib.pyplot as plt
from ray.rllib.algorithms.sac import SACConfig
from ray.tune.logger import pretty_print

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()],
)

# set Ray's logging level to error to suppress verbose output
logger = logging.getLogger('ray')
logger.setLevel(logging.ERROR)

@dataclass
class TrainingConfig:
    iterations: int = 10500
    eval_interval: int = 50
    checkpoint_frequency: int = 3600  
    early_stop_patience: int = 5000
    early_stop_min_improvement: float = 0.01
    num_workers: int = 8
    num_eval_workers: int = 4
    learning_rate: float = 3e-4
    grad_clip: float = 0.25
    tau: float = 0.005
    initial_alpha: float = 0.2
    batch_size: int = 512
    buffer_size: int = 500000
    hidden_layers: List[int] = None

    def __post_init__(self):
        self.hidden_layers = self.hidden_layers or [64, 64]

class TrainingManager:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.setup_directories()
        self.rewards: List[float] = []
        self.lengths: List[float] = []
        self.best_reward = float('-inf')
        self.patience_counter = 0
        
    def setup_directories(self) -> None:
        """Setup necessary directories"""
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.checkpoint_dir = os.path.join(self.script_dir, "checkpoints")
        self.gif_dir = os.path.join(self.script_dir, "policy_gifs")
        self.tensorboard_dir = os.path.join(self.script_dir, "tensorboard")
        
        for directory in [self.checkpoint_dir, self.gif_dir, self.tensorboard_dir]:
            os.makedirs(directory, exist_ok=True)

    def setup_sac_config(self) -> SACConfig:
        """Configure SAC algorithm for EnhancedGeosearchModel w/ normalized env."""
        from ray import tune

        # -- Remove or comment out old direct gym.register calls here, if any. --

        # Instead, register a new environment ID, "GeosearchEnvNorm-v0",
        # that returns NormalizeObservation(GeosearchEnv()).
        def env_creator(env_config):
            raw_env = GeosearchEnv()
            return NormalizeObservation(raw_env)

        tune.register_env("GeosearchEnvNorm-v0", env_creator)

        # Build the SAC config
        config = SACConfig()
        config.framework_str = "torch"
        config.env = "GeosearchEnvNorm-v0"
        
        # RLlib's "API stack" toggles
        config.api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False
        )
        
        # Resources
        config.num_workers = self.config.num_workers
        config.evaluation_num_env_runners = self.config.num_eval_workers
        config.evaluation_interval = self.config.eval_interval
        config.evaluation_duration = 30  # or your preference
        
        # Training parameters
        config.training(
            lr=self.config.learning_rate,
            tau=self.config.tau,
            initial_alpha=self.config.initial_alpha,
            train_batch_size=self.config.batch_size,
            policy_model_config={
                "custom_model": "geo_model",
                "custom_action_dist": "categorical",
            },
            q_model_config={
                "custom_model": "geo_model",
                "custom_action_dist": "categorical",
            },
            replay_buffer_config={
                "type": "ReplayBuffer",
                "capacity": self.config.buffer_size,
                "alpha": 0.6,
                "beta": 0.4
            }
        )
        
        # Model configuration
        config.model = {
            "_disable_preprocessor_api": True,
            "max_seq_len": 20
        }
        
        # Batch settings
        config.rollout_fragment_length = 200
        config.target_network_update_freq = 1
        
        return config

    def save_checkpoint(self, algo, is_best: bool = False) -> Optional[str]:
        """Save algorithm checkpoint"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prefix = "sac_lunar_best" if is_best else f"sac_lunar_{timestamp}"
            save_dir = os.path.join(self.checkpoint_dir, prefix)
            os.makedirs(save_dir, exist_ok=True)
            
            checkpoint_path = algo.save(save_dir)
            
            # Save training state
            state = {
                'rewards': self.rewards,
                'lengths': self.lengths,
                'best_reward': self.best_reward,
                'iteration': len(self.rewards) * self.config.eval_interval
            }
            with open(os.path.join(save_dir, 'training_state.json'), 'w') as f:
                json.dump(state, f)
                
            return checkpoint_path
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {str(e)}")
            return None

    def load_checkpoint(self, algo, checkpoint_dir: str) -> bool:
        """Load algorithm checkpoint"""
        try:
            # Find checkpoint file
            checkpoint_file = None
            for file in os.listdir(checkpoint_dir):
                if file.endswith('.pkl'):
                    checkpoint_file = os.path.join(checkpoint_dir, file)
                    break
            
            if checkpoint_file is None:
                logger.error(f"No checkpoint found in {checkpoint_dir}")
                return False
                
            # Load checkpoint
            algo.restore(checkpoint_dir)
            
            # Load training state if available
            state_file = os.path.join(checkpoint_dir, 'training_state.json')
            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    self.rewards = state['rewards']
                    self.lengths = state['lengths']
                    self.best_reward = state['best_reward']
                    
            logger.info("Successfully restored checkpoint")
            return True
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {str(e)}")
            return False

    def plot_metrics(self) -> None:
        """Plot and save training metrics"""
        plt.figure(figsize=(16, 8))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.rewards)
        plt.title('Mean Rewards')
        plt.xlabel('Evaluation Interval')
        plt.ylabel('Mean Reward')
        
        plt.subplot(1, 2, 2)
        plt.plot(self.lengths)
        plt.title('Mean Episode Length')
        plt.xlabel('Evaluation Interval')
        plt.ylabel('Episode Length')
        
        plt.savefig(os.path.join(self.script_dir, "training_progress.png"))
        plt.close()

    def check_early_stopping(self, current_reward: float) -> bool:
        """Check early stopping conditions"""
        if current_reward > self.best_reward + self.config.early_stop_min_improvement:
            self.best_reward = current_reward
            self.patience_counter = 0
            return False
        
        self.patience_counter += 1
        if self.patience_counter >= self.config.early_stop_patience:
            logger.info(f"Early stopping triggered after {self.patience_counter} evaluations without improvement")
            return True
        return False

    def train(self) -> None:
        """Main training loop"""
        try:
            # Initialize Ray
            ray.init()
            
            # Build and configure the SAC algorithm
            algo = self.setup_sac_config().build()
            
            # Training loop variables
            start_time = time.time()
            last_checkpoint_time = start_time
            
            # Main training loop
            for i in range(self.config.iterations):
                try:
                    result = algo.train()
                    
                    # Monitor gradients
                    if (i+1) % 10 == 0:
                        grad_norm = result.get("info", {}).get("grad_norm", 0)
                        if grad_norm > 10:
                            logger.warning(f"High gradient norm detected: {grad_norm}")
                    
                    # Evaluation
                    if (i+1) % self.config.eval_interval == 0:
                        self._handle_evaluation(algo, i, start_time)
                        
                        # Check early stopping
                        if self.check_early_stopping(self.rewards[-1]):
                            break
                    
                    # Periodic checkpoint
                    current_time = time.time()
                    if current_time - last_checkpoint_time > self.config.checkpoint_frequency:
                        self.save_checkpoint(algo)
                        last_checkpoint_time = current_time
                        
                except ValueError as e:
                    if "nan" in str(e).lower():
                        if not self._handle_nan_error(algo, i):
                            raise e
                    else:
                        raise e
                        
            # Final saves
            final_checkpoint_path = self.save_checkpoint(algo)
            logger.info(f"Training completed. Final checkpoint: {final_checkpoint_path}")
            
            # Visualize final policy
            # NOTE: We want the same normalization at inference:
            from lunabot.geosearch import GeosearchEnv, NormalizeObservation
            env = NormalizeObservation(GeosearchEnv(render_mode='human'))
            self.visualize_policy(algo, env)
            
        finally:
            ray.shutdown()

    def visualize_policy(self, algo, env, episodes: int = 3, max_steps: int = 100) -> None:
        """Visualize policy behavior"""
        try:
            import pygame
            pygame.init()
            
            # Setup policy wrapper
            policy_wrapper = RLlibPolicyWrapper(algo)
            
            if hasattr(env, 'render_mode') and env.render_mode != 'human':
                env.render_mode = 'human'
            if hasattr(env, '_init_render'):
                env._init_render()
            
            frames = []
            total_reward = 0
            
            for episode in range(episodes):
                state, info = env.reset()
                episode_reward = 0
                terminated = False
                truncated = False
                steps = 0
                
                while not terminated and not truncated and steps < max_steps:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            raise KeyboardInterrupt
                    
                    env.render()
                    if hasattr(env, 'screen'):
                        frame = pygame.surfarray.array3d(env.screen)
                        frames.append(np.transpose(frame, (1, 0, 2)))
                    
                    action = policy_wrapper.get_action(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    
                    episode_reward += reward
                    state = next_state
                    steps += 1
                    
                    pygame.time.wait(66)  # ~15 FPS
                    
                total_reward += episode_reward
                logger.info(f"Episode {episode + 1} finished with reward: {episode_reward:.2f}")
                logger.info(f"Episode {episode + 1} length: {steps} steps")
            
            # Save visualization as GIF
            gif_path = os.path.join(self.gif_dir, f"sac_policy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.gif")
            from lunabot.geosearch import Utils
            Utils.create_gif(frames, filename=gif_path, duration=66)
            logger.info(f"Policy visualization saved to: {gif_path}")
            
            avg_reward = total_reward / episodes
            logger.info(f"Average reward over {episodes} episodes: {avg_reward:.2f}")

        except KeyboardInterrupt:
            logger.info("\nVisualization interrupted by user")
        except Exception as e:
            logger.error(f"Visualization error: {str(e)}")
        finally:
            if hasattr(env, 'close'):
                env.close()
            pygame.quit()

    def _handle_evaluation(self, algo, iteration: int, start_time: float) -> None:
        """Handle evaluation phase"""
        evaluation_results = algo.evaluate()
        metrics = evaluation_results['env_runners']
        
        # Get rewards - prefer episode_return over episode_reward if available
        current_reward = metrics.get('episode_return_mean', metrics.get('episode_reward_mean', 0))
        episode_length = metrics.get('episode_len_mean', 0)
        num_episodes = metrics.get('episodes_this_iter', metrics.get('num_episodes', 0))
        
        # Get min/max rewards for additional insight
        reward_min = metrics.get('episode_return_min', metrics.get('episode_reward_min', 0))
        reward_max = metrics.get('episode_return_max', metrics.get('episode_reward_max', 0))
        
        self.rewards.append(current_reward)
        self.lengths.append(episode_length)
        
        # Save best model
        if current_reward > self.best_reward:
            self.best_reward = current_reward
            self.save_checkpoint(algo, is_best=True)
        
        # Progress report
        elapsed_time = (time.time() - start_time) / 3600
        logger.info(f"\nIteration {iteration+1}/{self.config.iterations}")
        logger.info(f"Time elapsed: {elapsed_time:.2f} hours")
        logger.info(f"Episodes this iteration: {num_episodes}")
        logger.info(f"Average episode length: {episode_length:.2f} steps")
        logger.info(f"Episode rewards - Min: {reward_min:.2f}, Mean: {current_reward:.2f}, Max: {reward_max:.2f}")
        logger.info(f"Best reward so far: {self.best_reward:.2f}")
        
        # Update plots
        self.plot_metrics()

    def _handle_nan_error(self, algo, iteration: int) -> bool:
        """Handle NaN errors with environment reset"""
        logger.warning(f"NaN detected at iteration {iteration+1}")
        best_checkpoint_dir = os.path.join(self.checkpoint_dir, "sac_lunar_best")
        
        if not os.path.exists(best_checkpoint_dir):
            logger.error("No checkpoint directory available")
            return False
            
        if not self.load_checkpoint(algo, best_checkpoint_dir):
            return False
            
        # Reset environment runners
        if hasattr(algo, 'workers') and algo.workers is not None:
            algo.workers.sync_weights()
            for worker in algo.workers.remote_workers():
                worker.reset.remote()
        
        if hasattr(algo, '_local_env_runner'):
            algo._local_env_runner.reset()
        
        logger.info("Successfully restored checkpoint and reset environments")
        time.sleep(1)
        return True

def main():
    config = TrainingConfig()
    trainer = TrainingManager(config)
    trainer.train()

if __name__ == "__main__":
    main()
