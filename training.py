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
from Moon_Rover import LunarRover3DEnv, NormalizeObservation

###############################
# 1) REGISTER THE ENVIRONMENT #
###############################
def lunar_rover_env_creator(env_config):
    """
    Factory function for RLlib to create your LunarRover3DEnv,
    then wrap it with NormalizeObservation.
    """
    dem_path = env_config.get("dem_path", "/Users/jbm/Desktop/Moon_Rover_SouthPole/src/map/LDEM_80S_20MPP_ADJ.tiff")
    subregion_window = env_config.get("subregion_window", None)
    
    raw_env = LunarRover3DEnv(
        dem_path=dem_path,
        subregion_window=subregion_window,
        max_slope_deg=25,
        smooth_sigma=None,
        desired_distance_m=30000,
        distance_reward_scale=0.08,
        cold_region_scale=30
    )
    return NormalizeObservation(raw_env)

register_env("LunarRover-v0", lunar_rover_env_creator)

###################################
# 2) TRAINING MANAGER CLASS       #
###################################
class TrainingManager:
    def __init__(self,
                 stop_iters=100,
                 checkpoint_dir="./checkpoints",
                 plot_dir="./training_plots"):
        self.stop_iters = stop_iters
        self.checkpoint_dir = checkpoint_dir
        self.plot_dir = plot_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)

        # Stats
        self.rewards = []
        self.lengths = []
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
                    "subregion_window": (0, 6000, 0, 6000),
                }
            )
            .framework("torch")
            .env_runners(
                num_env_runners=4  # ✅ OK
            )
            .evaluation(
                evaluation_num_workers=2,  # ✅ This is still separate in 2.6
                evaluation_interval=1,
                evaluation_config={"explore": False}
            )
            .training(
                train_batch_size=2048,
                replay_buffer_config={
                    "type": "MultiAgentPrioritizedReplayBuffer",
                    "capacity": 1000000,
                    "prioritized_replay_alpha": 0.6,  # default alpha (tunable)
                    "prioritized_replay_beta": 0.4,   # default beta (tunable)
                    "prioritized_replay_eps": 1e-6,   # small constant to avoid zero priority
                }
            )
        )

        algo = sac_config.build()

        #################################################
        # B) Setup live-plotting with matplotlib         #
        #    We'll create a figure and update it inline  #
        #################################################
        plt.ion()  # interactive mode on
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        live_plot_path = os.path.join(self.plot_dir, "training_progress_live.png")

        for i in range(self.stop_iters):
            ##########################
            # 1) Single train step   #
            ##########################
            result = algo.train()

            # RLlib metrics
            mean_reward = result["env_runners"]["episode_reward_mean"]
            mean_length = result["env_runners"]["episode_len_mean"]
            self.rewards.append(mean_reward)
            self.lengths.append(mean_length)

            ##########################################
            # 2) Checkpoint if we have a new best    #
            ##########################################
            if mean_reward > self.best_reward:
                self.best_reward = mean_reward
                best_ckpt_path = algo.save(self.checkpoint_dir)
                print(f"[{i}] New best reward={mean_reward:.3f}; checkpoint={best_ckpt_path}")

            ##########################
            # 3) Update live plots   #
            ##########################
            if (i + 1) % 10 == 0:
                ax1.clear()
                ax1.plot(self.rewards, label="Mean Reward")
                ax1.set_title("Episode Reward Mean")
                ax1.legend()

                ax2.clear()
                ax2.plot(self.lengths, label="Mean Ep. Length", color="orange")
                ax2.set_title("Episode Length Mean")
                ax2.legend()

                plt.pause(0.01)
                
                # Overwrite the same plot file
                plt.savefig(live_plot_path)  # <--- OVERWRITE SAME FILE
                print(f"Updated live plot at: {live_plot_path}")

            ##########################
            # 4) Log output updates #
            ##########################
            if (i + 1) % 50 == 0:
                print(f"Iter={i+1}, mean_reward={mean_reward:.3f}, length={mean_length:.3f}")

        # Turn off interactive mode
        plt.ioff()
        # Final save of the figure
        plt.savefig("training_progress.png")
        plt.close(fig)

        # Final checkpoint
        final_ckpt_path = algo.save(self.checkpoint_dir)
        print(f"\nTraining finished. Final checkpoint: {final_ckpt_path}")

        # Optionally record a short GIF of the final (or best) policy
        self.record_gif(algo, filename="lunar_rover_final.gif")

        ray.shutdown()

        final_plot_path = os.path.join(self.plot_dir, "training_progress_final.png")
        plt.savefig(final_plot_path)
        plt.close(fig)

    def record_gif(self, algo, filename="lunar_rover_final.gif", max_steps=3000):
        """
        Runs one test episode with `algo`, capturing frames from .render(save_frames=True).
        Saves an animated GIF to `filename`.
        """
        print(f"Recording a policy rollout to {filename} ...")
        env = lunar_rover_env_creator({})  # or pass same env_config
        obs, _ = env.reset()
        images = []

        done = False
        truncated = False
        steps = 0
        while not done and not truncated and steps < max_steps:
            # Inference
            action = algo.compute_single_action(obs, explore=False)
            obs, reward, done, truncated, info = env.step(action)
            
            # Render screenshot
            frame = env.render(mode='human', save_frames=True)
            if frame is not None:
                images.append(frame)
            steps += 1

        if images:
            imageio.mimsave(filename, images, fps=8)
            print(f"GIF saved to: {filename}")
        else:
            print("No frames were captured; GIF not saved.")
        

if __name__ == "__main__":
    tm = TrainingManager(stop_iters=300)
    tm.train()
    print("Training complete.")