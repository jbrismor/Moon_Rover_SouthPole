import ray
from ray.rllib.algorithms.sac import SACConfig
from ray.tune.registry import register_env
from Moon_Rover import LunarRover3DEnv
import numpy as np
import os
import pickle
from types import SimpleNamespace
import tempfile

def lunar_rover_env_creator_longer(env_config):
    return LunarRover3DEnv(
        dem_path="/Users/jbm/Desktop/Moon_Rover_SouthPole/src/map/LDEM_80S_20MPP_ADJ.tiff",
        subregion_window=(5000, 7000, 5000, 7000),
        max_slope_deg=25,
        smooth_sigma=None,
        desired_distance_m=20000,
        distance_reward_scale=0.8,
        step_penalty = -0.001,
        cold_region_scale=50,
        num_cold_regions=3,
        goal_radius_m=50,
        max_num_steps=10000,
        cold_penalty = -50.0,
        slope_penalty = -10.0,
        forward_speed = 10,
        cold_region_locations=[(29985, 10000)],
        goal_reward=5000
    )

register_env("LunarRoverLongDist-v0", lunar_rover_env_creator_longer)

def fix_checkpoint_config(checkpoint_path):
    with open(checkpoint_path, "rb") as f:
        data = pickle.load(f)
    # Remove problematic function references.
    if isinstance(data.get("config"), dict):
        data["config"].pop("custom_metrics_fn", None)
        # Also remove from nested evaluation_config if it exists.
        if "evaluation_config" in data["config"]:
            if isinstance(data["config"]["evaluation_config"], dict):
                data["config"]["evaluation_config"].pop("custom_metrics_fn", None)
        # Convert config dict to a SimpleNamespace for attribute access.
        data["config"] = SimpleNamespace(**data["config"])
    # Ensure the worker state has a "state" key.
    if "worker" in data:
        if isinstance(data["worker"], dict) and "state" not in data["worker"]:
            data["worker"]["state"] = {}
    fixed_path = os.path.join(tempfile.gettempdir(), "fixed_algorithm_state.pkl")
    with open(fixed_path, "wb") as f:
        pickle.dump(data, f)
    return fixed_path

def test_model(checkpoint_path, num_runs=3):
    # Initialize Ray
    ray.init()

    # Recreate the SAC configuration (using the same API stack as training)
    sac_config = (
        SACConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False
        )
        .environment(env="LunarRoverLongDist-v0")
        .framework("torch")
        .env_runners(num_env_runners=0)
        .training(
            model={"fcnet_hiddens": [256, 256]},
            replay_buffer_config={
                "type": "MultiAgentPrioritizedReplayBuffer",
                "capacity": 1000000,
                "prioritized_replay_alpha": 0.5,
                "prioritized_replay_beta": 0.7,
                "prioritized_replay_eps": 1e-6,
            }
        )
    )

    # Build algorithm
    algo = sac_config.build()
    
    # Fix the checkpoint config and restore
    fixed_checkpoint_path = fix_checkpoint_config(checkpoint_path)
    algo.restore(fixed_checkpoint_path)
    print(f"Loaded checkpoint from {fixed_checkpoint_path}")

    # Create environment instance
    env = lunar_rover_env_creator_longer({})

    # Run evaluation episodes
    for run in range(num_runs):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        success = False

        while not done:
            action = algo.compute_single_action(obs, explore=False)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            success = info.get("is_success", False)
            if steps >= 10000:
                break

        print(f"\nRun {run+1} Results:")
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Steps Taken: {steps}")
        print(f"Success: {'Yes' if success else 'No'}")
        print(f"Final Distance: {info.get('distance_covered', 0):.2f}m")
        print(f"Cold Zone Penalties: {info.get('cold_penalties', 0)}")
        print(f"Slope Penalties: {info.get('slope_penalties', 0)}")

    ray.shutdown()

if __name__ == "__main__":
    checkpoint_path = os.path.abspath("checkpoints_20km/algorithm_state.pkl")
    test_model(checkpoint_path, num_runs=3)