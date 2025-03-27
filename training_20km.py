import os
import numpy as np
import matplotlib.pyplot as plt
import ray
from ray.rllib.algorithms.sac import SACConfig
from ray.tune.registry import register_env

import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

from Moon_Rover import LunarRover3DEnv  

##################################################
# 1) REGISTER NEW ENV: "LunarRoverLongDist-v0" #
#    with a LONGER desired_distance_m            #
##################################################
def lunar_rover_env_creator_longer(env_config):
    """
    Example environment factory for a 'longer distance' scenario.
    Increase desired_distance_m from 1000 -> 2000, etc.
    """
    dem_path = env_config.get(
        "dem_path",
        "/Users/jbm/Desktop/Moon_Rover_SouthPole/src/map/LDEM_80S_20MPP_ADJ.tiff"
    )
    subregion_window = env_config.get("subregion_window", (5000, 7000, 5000, 7000))

    # New environment with increased desired_distance_m and new params
    return LunarRover3DEnv(
        dem_path=dem_path,
        subregion_window=subregion_window,
        max_slope_deg=25,
        smooth_sigma=None,
        desired_distance_m=20000,  # <--- Increased distance
        distance_reward_scale=0.15,
        cold_region_scale=50,
        num_cold_regions=3,
        max_num_steps=5000,
        forward_speed = 10.0,
        step_penalty =  -0.001,
        cold_penalty = -100.0,
        slope_penalty = -10.0,
        goal_radius_m=50,
        cold_region_locations=[(29985, 10000)]
    )

# Register the longer-distance variant under new name
register_env("LunarRoverLongDist-v0", lunar_rover_env_creator_longer)


##################################################
# 2) CONTINUE TRAINING FUNCTION                  #
#    (Restores from single-folder checkpoint)    #
##################################################
def continue_training(
    checkpoint_path,
    stop_iters=10000,
    new_checkpoint_dir="./checkpoints_continued"
):
    """
    Loads the SAC algorithm from an existing single-folder checkpoint
    (containing algorithm_state.pkl, etc.) and continues training on
    a new environment configuration (longer distance).
    """

    # 1) Initialize Ray
    ray.init()

    # 2) SAME SACConfig from your original training,
    #    but point it at the new environment ("LunarRoverLongDist-v0").
    sac_config = (
        SACConfig()
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .environment(
            env="LunarRoverLongDist-v0",
            env_config={
                "dem_path": "/Users/jbm/Desktop/Moon_Rover_SouthPole/src/map/LDEM_80S_20MPP_ADJ.tiff",
                "subregion_window": (5000, 7000, 5000, 7000)
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
            evaluation_config={"explore": False}
        )
    )

    # 3) Build the SAC algorithm object
    algo = sac_config.build()

    # 4) Restore from the existing single-folder checkpoint
    print(f"Loading from checkpoint folder: {checkpoint_path}")
    algo.restore(checkpoint_path)

    print("Successfully loaded the previous policy weights.\n")

    # 5) Continue training
    rewards, lengths = [], []
    best_reward = float("-inf")
    os.makedirs(new_checkpoint_dir, exist_ok=True)

    # Initialize live plotting
    plt.ion()  # Enable interactive mode
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    live_plot_path = "continued_training_progress_live.png"
    final_plot_path = "continued_training_progress_final.png"

    for i in range(stop_iters):
        result = algo.train()
        mean_reward = result["env_runners"]["episode_reward_mean"]
        mean_length = result["env_runners"]["episode_len_mean"]
        rewards.append(mean_reward)
        lengths.append(mean_length)

        # Checkpointing
        if mean_reward > best_reward:
            best_reward = mean_reward
            ckpt_path = algo.save(new_checkpoint_dir)
            print(f"[Iter={i}] New best reward={mean_reward:.3f}; checkpoint={ckpt_path}")

        # Live plotting and logging
        if (i + 1) % 10 == 0:  # Update every 10 iterations
            # Update plots
            ax1.clear()
            ax1.plot(rewards, label="Mean Reward")
            ax1.set_title("Continued Training: Reward Progress")
            ax1.legend()

            ax2.clear()
            ax2.plot(lengths, label="Episode Length", color="orange")
            ax2.set_title("Continued Training: Length Progress")
            ax2.legend()

            plt.pause(0.01)  # Needed for live updates
            plt.savefig(live_plot_path)
            print(f"Updated live plot: {live_plot_path}")

            # Console logging
            print(f"Iter={i+1}, mean_reward={mean_reward:.3f}, length={mean_length:.3f}")

    # Final cleanup and saving
    plt.ioff()
    plt.savefig(final_plot_path)
    plt.close(fig)
    print(f"Final training plot saved to: {final_plot_path}")

    # Final checkpoint
    final_ckpt_path = algo.save(new_checkpoint_dir)
    print(f"\nContinued training finished. Final checkpoint: {final_ckpt_path}")

    # Shutdown Ray
    ray.shutdown()


###################################
# 3) SCRIPT ENTRY POINT           #
###################################
if __name__ == "__main__":
    # This is the directory containing `algorithm_state.pkl`, `rllib_checkpoint.json`,
    my_checkpoint_path = "/Users/jbm/Desktop/Moon_Rover_SouthPole/checkpoints"

    # Run additional training
    continue_training(
        checkpoint_path=my_checkpoint_path,
        stop_iters=10000,
        new_checkpoint_dir="./checkpoints_continued"
    )

    print("Extended training complete.")
