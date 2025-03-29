import ray
from ray.rllib.algorithms.algorithm import Algorithm
import pyvista as pv
from Moon_Rover import LunarRover3DEnv
import os
from ray.tune.registry import register_env

def env_creator(config):

    dem_path = config.get("dem_path", "/Users/jbm/Desktop/Moon_Rover_SouthPole/src/map/LDEM_80S_20MPP_ADJ.tiff")
    subregion_window = config.get("subregion_window", None)

    return LunarRover3DEnv(
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

register_env("LunarRover-v0", env_creator)

def test_policy(checkpoint_path):
    # Initialize plotter and environment
    plotter = pv.Plotter()
    env = env_creator({})
    env.set_plotter(plotter)
    
    # Load trained policy
    algo = Algorithm.from_checkpoint(checkpoint_path)
    policy = algo.get_policy()

    for episode in range(3):
        obs, _ = env.reset(options={"record_path": False})
        done = False
        terminated = False
        outcome = "Timeout"  # Default outcome
        
        print(f"\n=== Episode {episode+1} ===")
        
        while not done:
            action = policy.compute_single_action(obs, explore=False)[0]
            obs, reward, done, terminated, _ = env.step(action)
            
            if terminated:
                outcome = "Crash" if reward < 0 else "Success"
        
        # Store path with outcome
        if env.current_path:
            env.agent_paths.append({
                'points': env.current_path.copy(),
                'color': env.path_colors[outcome],
                'outcome': outcome
            })
    
    # Final render with all paths
    env.render(show_path=True)
    # plotter.show()  # Show the window after all episodes

if __name__ == "__main__":
    try:
        ray.init()
        test_policy(checkpoint_path=os.path.abspath("./checkpoints"))
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        ray.shutdown()