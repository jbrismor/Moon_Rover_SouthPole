# training.py

import os
import numpy as np
import matplotlib.pyplot as plt
import ray
print(f"Ray version: {ray.__version__}")

from ray import tune
from ray.rllib.algorithms.sac import SACConfig
from ray.tune.registry import register_env
import torch
import imageio
import time
from ray.rllib.models import ModelCatalog

import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

# Import your environment code
from Moon_Rover import LunarRover3DEnv # NormalizeObservation

###############################
# 1) REGISTER THE ENVIRONMENT #
###############################
def lunar_rover_env_creator(env_config):
    """
    Factory function for RLlib to create your LunarRover3DEnv
    """
    dem_path = env_config.get("dem_path", "/Users/jbm/Desktop/Moon_Rover_SouthPole/src/map/LDEM_80S_20MPP_ADJ.tiff")
    subregion_window = env_config.get("subregion_window", None)
    
    raw_env = LunarRover3DEnv(
        dem_path=dem_path,
        subregion_window=subregion_window,
        max_slope_deg=25,
        smooth_sigma=None,
        desired_distance_m=1000,
        distance_reward_scale=0.5,
        step_penalty = -0.01,
        cold_region_scale=10,
        num_cold_regions=1,
        goal_radius_m=50,
        max_num_steps=600,
        cold_penalty = -100.0,
        slope_penalty = -20.0,
        forward_speed = 10
    )

    return raw_env #NormalizeObservation(raw_env)

register_env("LunarRover-v0", lunar_rover_env_creator)

# beta_anneal_callback.py

from beta_anneal_callback import BetaAnnealCallback

# Add this custom metrics function near the top of training.py
from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes

def custom_metrics_fn(episodes):
    """Custom function to collect 'is_success' from episode info."""
    return {
        "is_success": [
            bool(episode.last_info_for().get("is_success", False))
            for episode in episodes
        ]
    }

###################################
# 2) TRAINING MANAGER CLASS       #
###################################
class TrainingManager:
    def __init__(self,
                 stop_iters=1300,
                 checkpoint_dir="./checkpoints",
                 plot_dir="./training_plots"):
        self.stop_iters = stop_iters
        self.checkpoint_dir = checkpoint_dir
        self.plot_dir = plot_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)

        # Stats
        self.train_rewards = []  # Track training rewards
        self.eval_rewards = []   # Track evaluation rewards
        self.train_lengths = []  # Corrected: Track training episode lengths
        self.eval_lengths = []     # Track evaluation episode lengths
        self.success_rates = []    # Track percentage of successful eval episodes
        self.eval_iterations = []  # Track when evaluations happened
        self.best_reward = float("-inf")

    def train(self):
        ###############################################
        # A) Initialize Ray and Build the SAC trainer #
        ###############################################
        ray.init()

        sac_config = (
            SACConfig()
            .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
            .environment(
                env="LunarRover-v0",
                env_config={
                    "dem_path": "/Users/jbm/Desktop/Moon_Rover_SouthPole/src/map/LDEM_80S_20MPP_ADJ.tiff",
                    "subregion_window": (6000, 7000, 6000, 7000)
                }
            )
            .framework("torch")
            .env_runners(
                num_env_runners=4,
                rollout_fragment_length=600,
                exploration_config={ 
                    "type": "StochasticSampling",
                    # "initial_epsilon": 1.0,
                    # "final_epsilon": 0.1,
                    # "epsilon_timesteps": 100000
                }
            )
            .training(
                train_batch_size=2048,
                gamma=0.99,
                tau=0.005,
                policy_model_config={
                "fcnet_hiddens": [256, 256], # [512, 512] can also be a good choice
                "fcnet_activation": "swish",
                "use_layer_norm": False,  # Use layer normalization for better stability
                },
                q_model_config={
                    "fcnet_hiddens": [256, 256],
                    "fcnet_activation": "swish",
                    "use_layer_norm": False,  # Use layer normalization for better stability
                },
                # policy_model_config={
                #     "custom_model": "terrain_policy_model",
                # },
                # q_model_config={
                #     "custom_model": "terrain_q_model",
                # },
                # model={  # Replace policy/q_model_config with this
                # "custom_model": "terrain_network",
                # "custom_model_config": {
                #     "q_arch": [512, 256, 512],
                #     "policy_arch": [512, 512],
                #     "use_layer_norm": True
                # }},
                # REMOVED: reward_scaling=0.1
                optimization_config={
                    "actor_learning_rate": 3e-4,
                    "critic_learning_rate": 3e-4,
                    "entropy_learning_rate": 1e-4,
                },
                replay_buffer_config={
                    "type": "MultiAgentPrioritizedReplayBuffer",
                    "capacity": 500000,
                    "alpha": 0.6,
                    "beta": 0.4,
                    "epsilon": 1e-6,
                },
                num_steps_sampled_before_learning_starts=10000,
                target_entropy="auto",
                n_step=5
            )
            .evaluation(
                evaluation_num_env_runners=2,
                evaluation_interval=1,
                evaluation_duration=20,
                # evaluation_parallel_to_training=True,
                evaluation_config={"explore": False, 
                                   "metrics_smoothing_episodes": 0,
                                   "custom_metrics_fn": custom_metrics_fn}
            ).callbacks(BetaAnnealCallback) 
        )

        # sac_config.rollouts(batch_mode="truncate_episodes")

        # sac_config.horizon = 500

        algo = sac_config.build()

        #################################################
        # B) Setup live-plotting with matplotlib         #
        #################################################
        # Initialize plotting
        plt.ion()
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        live_plot_path = os.path.join(self.plot_dir, "training_progress_live.png")

        for i in range(self.stop_iters):
            result = algo.train()

            # Track training metrics
            self.train_rewards.append(result["env_runners"]["episode_reward_mean"])
            self.train_lengths.append(result["env_runners"]["episode_len_mean"])

            # Track evaluation metrics if available
            if "evaluation" in result:
                eval_result = result["evaluation"]
                eval_metrics = eval_result.get("env_runners", {})
                
                eval_reward = eval_metrics.get("episode_reward_mean", 0.0)
                eval_length = eval_metrics.get("episode_len_mean", 0.0)
                
                # CORRECTED: Get success info from custom_metrics instead of hist_stats
                success_info = eval_metrics.get("custom_metrics", {}).get("is_success", [])
                
                self.eval_rewards.append(eval_reward)
                self.eval_lengths.append(eval_length)
                self.eval_iterations.append(i)
                self.success_rates.append(np.mean(success_info) if success_info and len(success_info) > 0 else 0.0
)


            # Update plots every 10 iterations
            if (i + 1) % 10 == 0:
                self._update_plots(ax1, ax2, ax3, ax4, live_plot_path)

            # Checkpointing and logging
            current_reward = self.eval_rewards[-1] if self.eval_rewards else self.train_rewards[-1]
            if current_reward > self.best_reward:
                self.best_reward = current_reward
                algo.save(self.checkpoint_dir)

            if (i + 1) % 50 == 0:
                print(f"Iter={i+1}, Train Reward={self.train_rewards[-1]:.1f}, "
                      f"Eval Reward={self.eval_rewards[-1] if self.eval_rewards else 'N/A':.1f}")

        # Final cleanup
        plt.ioff()
        plt.savefig(os.path.join(self.plot_dir, "training_progress_final.png"))
        plt.close(fig)
        ray.shutdown()

    def _update_plots(self, ax1, ax2, ax3, ax4, live_plot_path):
        """Update all four plot panels"""
        # Plot 1: Reward Comparison
        ax1.clear()
        ax1.plot(self.train_rewards, label="Train Reward", color="blue")
        if self.eval_rewards:
            ax1.plot(self.eval_iterations, self.eval_rewards, 
                    label="Eval Reward", linestyle="--", color="green", marker="o")
        ax1.set_title("Reward Progress")
        ax1.legend()

        # Plot 2: Training Episode Lengths
        ax2.clear()
        ax2.plot(self.train_lengths, label="Train Length", color="orange")
        ax2.set_title("Training Episode Lengths")
        ax2.legend()

        # Plot 3: Evaluation Episode Lengths
        ax3.clear()
        if self.eval_iterations:
            ax3.plot(self.eval_iterations, self.eval_lengths, 
                    label="Eval Length", color="red", marker="x")
        ax3.set_title("Evaluation Episode Lengths")
        ax3.legend()

        # Plot 4: Success Rate
        ax4.clear()
        if self.eval_iterations:
            ax4.plot(self.eval_iterations, self.success_rates,
                    label="Success Rate", color="purple", marker="s")
            ax4.set_ylim(0, 1.0)
        ax4.set_title("Evaluation Success Rate")
        ax4.legend()

        plt.pause(0.01)
        plt.savefig(live_plot_path)
        print(f"Updated live plot at: {live_plot_path}")

if __name__ == "__main__":
    tm = TrainingManager(stop_iters=1300)
    tm.train()
    print("Training complete.")