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
from ray.rllib.models import ModelCatalog
# Import your environment code
from Moon_Rover import LunarRover3DEnv # NormalizeObservation
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from beta_anneal_callback_100km import beta_anneal_callback_100km
from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes

def custom_metrics_fn(episodes):
    """Custom function to collect 'is_success' from episode info."""
    return {
        "is_success": [
            bool(episode.last_info_for().get("is_success", False))
            for episode in episodes
        ]
    }

##################################################
# 1) REGISTER NEW ENV: "LunarRoverLongDist-v0" #
#    with a LONGER desired_distance_m            #
##################################################
def lunar_rover_env_creator_longer100(env_config):
    """
    Example environment factory for a 'longer distance' scenario.
    Increase desired_distance_m from 1000 -> 2000, etc.
    """
    dem_path = env_config.get(
        "dem_path",
        "/Users/jbm/Desktop/Moon_Rover_SouthPole/src/map/LDEM_80S_20MPP_ADJ.tiff"
    )
    subregion_window = env_config.get("subregion_window", (4000, 9000, 4000, 9000))

    # New environment with increased desired_distance_m and new params
    return LunarRover3DEnv(
        dem_path=dem_path,
        subregion_window=subregion_window,
        max_slope_deg=25,
        smooth_sigma=None,
        desired_distance_m=100000,
        distance_reward_scale=1.25,
        step_penalty = -0.01,
        cold_region_scale=100,
        num_cold_regions=3,
        goal_radius_m=50,
        max_num_steps=100000,
        cold_penalty = -150.0,
        slope_penalty = -100.0,
        forward_speed = 10,
        cold_region_locations=[(29985, 10000)],
        goal_reward=100000
    )

# Register the longer-distance variant under new name
register_env("LunarRoverLongDist-vlong", lunar_rover_env_creator_longer100)


##################################################
# 2) CONTINUE TRAINING FUNCTION                  #
#    (Restores from single-folder checkpoint)    #
##################################################
def continue_training(
    checkpoint_path,
    stop_iters=10000,
    new_checkpoint_dir="./checkpoints_100km"
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
            env="LunarRoverLongDist-vlong",
            env_config={
                "dem_path": "/Users/jbm/Desktop/Moon_Rover_SouthPole/src/map/LDEM_80S_20MPP_ADJ.tiff",
                "subregion_window": (4000, 9000, 4000, 9000)
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
                "fcnet_hiddens": [256, 256, 256], # [512, 512] can also be a good choice
                "fcnet_activation": "swish",
                "use_layer_norm": False,  # Use layer normalization for better stability
                },
                q_model_config={
                    "fcnet_hiddens": [256, 256, 256],
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
                    "actor_learning_rate": 1e-4,
                    "critic_learning_rate": 1e-4,
                    "entropy_learning_rate": 1e-4,
                },
                replay_buffer_config={
                    "type": "MultiAgentPrioritizedReplayBuffer",
                    "capacity": 1000000,
                    "alpha": 0.6,
                    "beta": 0.2, # maybe increase
                    "epsilon": 1e-6,
                },
                num_steps_sampled_before_learning_starts=50000, # maybe increase to 50000
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
            ).callbacks(beta_anneal_callback_100km) # maybe increase the steps more for the callback
        )

    # 3) Build the SAC algorithm object
    algo = sac_config.build()

    # 4) Restore from the existing single-folder checkpoint
    print(f"Loading from checkpoint folder: {checkpoint_path}")
    algo.restore(checkpoint_path)

    print("Successfully loaded the previous policy weights.\n")

    # Training metrics storage
    train_rewards = []
    eval_rewards = []
    train_lengths = []
    eval_lengths = []
    success_rates = []
    eval_iterations = []
    best_reward = float("-inf")
    os.makedirs(new_checkpoint_dir, exist_ok=True)

    # Initialize plotting with 3 subplots
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    live_plot_path = "100km_training_progress_live.png"
    final_plot_path = "100km_training_progress_final.png"

    for i in range(stop_iters):
        result = algo.train()
        
        # Track training metrics
        train_rewards.append(result["env_runners"]["episode_reward_mean"])
        train_lengths.append(result["env_runners"]["episode_len_mean"])

        # Track evaluation metrics if available
        if "evaluation" in result:
            eval_result = result["evaluation"]
            eval_metrics = eval_result.get("env_runners", {})
            
            # Store evaluation metrics
            eval_reward = eval_metrics.get("episode_reward_mean", 0.0)
            eval_length = eval_metrics.get("episode_len_mean", 0.0)
            success_info = eval_metrics.get("custom_metrics", {}).get("is_success", [])
            
            eval_rewards.append(eval_reward)
            eval_lengths.append(eval_length)
            eval_iterations.append(i)
            success_rates.append(
                np.mean(success_info) 
                if success_info and len(success_info) > 0 
                else 0.0
            )

        # Checkpointing
        current_reward = eval_rewards[-1] if eval_rewards else train_rewards[-1]
        if current_reward > best_reward:
            best_reward = current_reward
            ckpt_path = algo.save(new_checkpoint_dir)
            print(f"[Iter={i}] New best reward={current_reward:.3f}; checkpoint={ckpt_path}")

        # Update plots every 10 iterations
        if (i + 1) % 10 == 0:
            ax1.clear()
            ax1.plot(train_rewards, label="Train Reward")
            if eval_rewards:
                ax1.plot(eval_iterations, eval_rewards, label="Eval Reward", linestyle="--")
            ax1.set_title("Reward Progress")
            ax1.legend()

            ax2.clear()
            ax2.plot(train_lengths, label="Train Length", color="blue")
            if eval_iterations:
                ax2.plot(eval_iterations, eval_lengths, label="Eval Length", color="orange", linestyle="--")
            ax2.set_title("Episode Lengths")
            ax2.legend()

            # ax3.clear()
            # if eval_iterations:
            #     ax3.plot(eval_iterations, success_rates, label="Success Rate", color="purple")
            #     ax3.set_ylim(0, 1.0)
            # ax3.set_title("Success Rate Progress")
            # ax3.legend()

            plt.pause(0.01)
            plt.savefig(live_plot_path)
            print(f"Iter={i+1}: Updated live plot")

    # Final cleanup and saving
    plt.ioff()
    plt.savefig(final_plot_path)
    plt.close(fig)
    
    # Final checkpoint
    final_ckpt_path = algo.save(new_checkpoint_dir)
    ray.shutdown()
    print(f"Final checkpoint: {final_ckpt_path}")


###################################
# 3) SCRIPT ENTRY POINT           #
###################################
if __name__ == "__main__":
    # This is the directory containing `algorithm_state.pkl`, `rllib_checkpoint.json`,
    my_checkpoint_path = "/Users/jbm/Desktop/Moon_Rover_SouthPole/checkpoints_20km"

    # Run additional training
    continue_training(
        checkpoint_path=my_checkpoint_path,
        stop_iters=10000,
        new_checkpoint_dir="./checkpoints_100km"
    )

    print("Extended training complete.")