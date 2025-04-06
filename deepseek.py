import ray
from ray.rllib.algorithms.sac import SACConfig
from ray.tune.registry import register_env
from Moon_Rover import LunarRover3DEnv
import numpy as np
import os
import pickle
import tempfile

# Define an AttrDict to allow attribute access on dictionary keys.
class AttrDict(dict):
    def __getattr__(self, key):
        try:
            value = self[key]
            if isinstance(value, dict):
                return AttrDict(value)
            return value
        except KeyError:
            raise AttributeError(key)
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

# Environment creation function (must match training exactly)
def lunar_rover_env_creator_longer(env_config):
    return LunarRover3DEnv(
        dem_path="/Users/jbm/Desktop/Moon_Rover_SouthPole/src/map/LDEM_80S_20MPP_ADJ.tiff",
        subregion_window=(5000, 7000, 5000, 7000),
        max_slope_deg=25,
        smooth_sigma=None,
        desired_distance_m=20000,
        distance_reward_scale=0.8,
        step_penalty=-0.001,      # Make sure these match training!
        cold_region_scale=50,
        num_cold_regions=3,
        goal_radius_m=50,
        max_num_steps=10000,
        cold_penalty=-50.0,
        slope_penalty=-10.0,
        forward_speed=10,
        cold_region_locations=[(29985, 10000)],
        goal_reward=5000
    )

# Register the environment with Ray
register_env("LunarRoverLongDist-v0", lunar_rover_env_creator_longer)

def fix_checkpoint_config(checkpoint_path):
    """
    Load the checkpoint pickle, remove problematic function references,
    and convert the "config" dict to an AttrDict for attribute access.
    Also, ensure a "state" key exists in the worker section.
    """
    with open(checkpoint_path, "rb") as f:
        data = pickle.load(f)
    # Remove any problematic function references from the config.
    if isinstance(data.get("config"), dict):
        data["config"].pop("custom_metrics_fn", None)
        if "evaluation_config" in data["config"]:
            if isinstance(data["config"]["evaluation_config"], dict):
                data["config"]["evaluation_config"].pop("custom_metrics_fn", None)
        # Convert the config dict to an AttrDict
        data["config"] = AttrDict(data["config"])
    # Patch the worker state to include a "state" key if missing.
    if "worker" in data:
        if isinstance(data["worker"], dict) and "state" not in data["worker"]:
            data["worker"]["state"] = {}
    fixed_path = os.path.join(tempfile.gettempdir(), "fixed_algorithm_state.pkl")
    with open(fixed_path, "wb") as f:
        pickle.dump(data, f)
    return fixed_path

def test_model(checkpoint_path, num_runs=3):
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    # Recreate the SAC configuration (matching training settings)
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
    
    # Build the algorithm
    algo = sac_config.build()
    
    # Fix the checkpoint's config and restore from it.
    fixed_checkpoint_path = fix_checkpoint_config(checkpoint_path)
    algo.restore(fixed_checkpoint_path)
    print(f"Loaded checkpoint from {fixed_checkpoint_path}")
    
    # Create an instance of the environment.
    env = lunar_rover_env_creator_longer({})
    
    # Run evaluation episodes.
    for run in range(num_runs):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        success = False
        
        while not done and steps < 10000:
            action = algo.compute_single_action(obs, explore=False)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            success = info.get("is_success", False)
            
        print(f"\nRun {run+1} Results:")
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Steps Taken: {steps}")
        print(f"Success: {'Yes' if success else 'No'}")
        print(f"Final Distance: {info.get('distance_covered', 0):.2f}m")
        print(f"Cold Zone Penalties: {info.get('cold_penalties', 0)}")
        print(f"Slope Penalties: {info.get('slope_penalties', 0)}")
    
    ray.shutdown()

if __name__ == "__main__":
    # Use the absolute path to the algorithm checkpoint file.
    checkpoint_path = os.path.abspath("checkpoints_20km/algorithm_state.pkl")
    test_model(checkpoint_path, num_runs=3)



# import os
# import ray
# import logging
# from ray.rllib.algorithms.sac import SACConfig
# from ray.tune.registry import register_env
# from Moon_Rover import LunarRover3DEnv
# from metrics import custom_metrics_fn  # Import from shared module

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[logging.StreamHandler()],
# )

# def lunar_rover_env_creator_longer(env_config):
#     return LunarRover3DEnv(
#         dem_path="/Users/jbm/Desktop/Moon_Rover_SouthPole/src/map/LDEM_80S_20MPP_ADJ.tiff",
#         subregion_window=(5000, 7000, 5000, 7000),
#         desired_distance_m=20000,
#         # Keep other parameters identical to training
#     )

# register_env("LunarRoverLongDist-v0", lunar_rover_env_creator_longer)

# class PolicyWrapper:
#     """Wrapper for RLlib policies with proper serialization"""
#     def __init__(self, algo):
#         self.algo = algo
    
#     def get_action(self, state):
#         return self.algo.compute_single_action(
#             observation=state,
#             explore=False
#         )

# def setup_sac_config():
#     """Create SAC config matching training setup"""
#     config = (
#         SACConfig()
#         .api_stack(
#             enable_rl_module_and_learner=False,
#             enable_env_runner_and_connector_v2=False
#         )
#         .environment(env="LunarRoverLongDist-v0")
#         .framework("torch")
#         .training(
#             train_batch_size=2048,
#             policy_model_config={"fcnet_hiddens": [256, 256]},
#             q_model_config={"fcnet_hiddens": [256, 256]},
#             replay_buffer_config={
#                 "type": "MultiAgentPrioritizedReplayBuffer",
#                 "capacity": 1000000,
#                 "alpha": 0.5,
#                 "beta": 0.7,
#                 "epsilon": 1e-6,
#             }
#         )
#         .evaluation(
#             evaluation_config={
#                 "explore": False,
#                 "custom_metrics_fn": custom_metrics_fn  # Use imported function
#             }
#         )
#     )
#     return config

# def test_model(checkpoint_path, num_runs=3):
#     ray.init(ignore_reinit_error=True)
    
#     try:
#         config = setup_sac_config()
#         algo = config.build()
        
#         # Resolve checkpoint path
#         if os.path.isdir(checkpoint_path):
#             # Find all checkpoint directories
#             checkpoint_dirs = [d for d in os.listdir(checkpoint_path) 
#                               if d.startswith("checkpoint_")]
            
#             if not checkpoint_dirs:
#                 raise ValueError(f"No checkpoints found in {checkpoint_path}")
            
#             # Sort by numerical value (checkpoint_000100, etc.)
#             checkpoint_dirs.sort(key=lambda x: int(x.split("_")[1]))
#             latest_checkpoint = checkpoint_dirs[-1]
#             checkpoint_path = os.path.join(checkpoint_path, latest_checkpoint)
        
#         logging.info(f"Loading from {checkpoint_path}")
        
#         # Verify checkpoint exists
#         if not os.path.exists(checkpoint_path):
#             raise FileNotFoundError(f"Checkpoint path {checkpoint_path} does not exist")
            
#         algo.restore(checkpoint_path)
        
#         # Test runs
#         env = lunar_rover_env_creator_longer({})
#         policy = PolicyWrapper(algo)
        
#         for run in range(num_runs):
#             obs, _ = env.reset()
#             done = False
#             total_reward = 0
#             steps = 0
            
#             while not done and steps < 10000:
#                 action = policy.get_action(obs)
#                 obs, reward, done, _, info = env.step(action)
#                 total_reward += reward
#                 steps += 1
                
#             logging.info(
#                 f"Run {run+1}: Reward={total_reward:.2f}, "
#                 f"Steps={steps}, Success={info.get('is_success', False)}"
#             )
            
#     finally:
#         ray.shutdown()

# if __name__ == "__main__":
#     test_model(
#         checkpoint_path="/Users/jbm/Desktop/Moon_Rover_SouthPole/checkpoints_20km",
#         num_runs=3
#     )