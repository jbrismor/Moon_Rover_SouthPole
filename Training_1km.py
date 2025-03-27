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
        distance_reward_scale=0.15,
        step_penalty = 0,
        cold_region_scale=10,
        num_cold_regions=1,
        goal_radius_m=50,
        max_num_steps=600,
        cold_penalty = -10.0,
        slope_penalty = -10.0,
        forward_speed = 10
    )
    return raw_env #NormalizeObservation(raw_env)

register_env("LunarRover-v0", lunar_rover_env_creator)

###################################
# 2) TRAINING MANAGER CLASS       #
###################################
class TrainingManager:
    def __init__(self,
                 stop_iters=3750,
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
        self.eval_iterations = []  # Track iterations where evaluation occurred
        self.lengths = []  # Track episode lengths
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
                rollout_fragment_length=600
            )
            .training(
                train_batch_size=2048,
                gamma=0.995,
                tau=0.005,
                policy_model_config={
                    "fcnet_hiddens": [256, 256, 256], # [512, 512] can also be a good choice
                    "fcnet_activation": "relu",
                },
                q_model_config={
                    "fcnet_hiddens": [256, 256, 256],
                    "fcnet_activation": "relu",
                },
                # REMOVED: reward_scaling=0.1
                optimization_config={
                    "actor_learning_rate": 3e-4,
                    "critic_learning_rate": 3e-4,
                    "entropy_learning_rate": 3e-4,
                },
                replay_buffer_config={
                    "type": "MultiAgentPrioritizedReplayBuffer",
                    "capacity": 1000000,
                    "alpha": 0.7,
                    "beta": 0.5,
                    "epsilon": 1e-6,
                },
                num_steps_sampled_before_learning_starts=10000,
                target_entropy="auto",
                n_step=3
            )
            .evaluation(
                evaluation_num_env_runners=2,
                evaluation_interval=5,
                # evaluation_parallel_to_training=True,
                evaluation_config={"explore": False}
            )
        )

        # sac_config.rollouts(batch_mode="truncate_episodes")

        # sac_config.horizon = 500

        algo = sac_config.build()

        #################################################
        # B) Setup live-plotting with matplotlib         #
        #################################################
        plt.ion()  # Interactive mode on
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        live_plot_path = os.path.join(self.plot_dir, "training_progress_live.png")

        for i in range(self.stop_iters):
            ##########################
            # 1) Single train step   #
            ##########################
            result = algo.train()

            # Extract metrics
            train_reward = result["env_runners"]["episode_reward_mean"]
            train_length = result["env_runners"]["episode_len_mean"]
            eval_reward = result.get("evaluation", {}).get("episode_reward_mean", None)
            
            # Track metrics
            self.train_rewards.append(train_reward)
            self.lengths.append(train_length)
            
            if eval_reward is not None:
                self.eval_rewards.append(eval_reward)
                self.eval_iterations.append(i)  # Track evaluation iteration

            ##########################################
            # 2) Checkpoint if we have a new best    #
            ##########################################
            current_reward = eval_reward if eval_reward is not None else train_reward
            if current_reward > self.best_reward:
                self.best_reward = current_reward
                best_ckpt_path = algo.save(self.checkpoint_dir)
                print(f"[{i}] New best reward={current_reward:.3f}; checkpoint={best_ckpt_path}")

            ##########################
            # 3) Update live plots   #
            ##########################
            if (i + 1) % 10 == 0:
                ax1.clear()
                ax1.plot(self.train_rewards, label="Train Reward")
                if self.eval_rewards:
                    ax1.plot(self.eval_iterations, self.eval_rewards, 
                            label="Eval Reward", linestyle="--", marker="o")
                ax1.set_title("Episode Reward Mean")
                ax1.legend()

                ax2.clear()
                ax2.plot(self.lengths, label="Mean Ep. Length", color="orange")
                ax2.set_title("Episode Length Mean")
                ax2.legend()

                plt.pause(0.01)
                plt.savefig(live_plot_path)
                print(f"Updated live plot at: {live_plot_path}")

            ##########################
            # 4) Log output updates  #
            ##########################
            if (i + 1) % 50 == 0:
                print(f"Iter={i+1}, reward={current_reward:.3f}, length={train_length:.3f}")

        # Turn off interactive mode and save final plot
        plt.ioff()
        plt.savefig(os.path.join(self.plot_dir, "training_progress_final.png"))
        plt.close(fig)

        # Final checkpoint
        final_ckpt_path = algo.save(self.checkpoint_dir)
        print(f"\nTraining finished. Final checkpoint: {final_ckpt_path}")

        ray.shutdown()

if __name__ == "__main__":
    tm = TrainingManager(stop_iters=3750)
    tm.train()
    print("Training complete.")