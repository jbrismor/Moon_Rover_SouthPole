from ray.rllib.algorithms.callbacks import DefaultCallbacks

class BetaAnnealCallback(DefaultCallbacks):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initial_beta = 0.4
        self.final_beta = 1.0
        self.anneal_timesteps = 1_000_000  # Over how many timesteps to go from initial -> final

    def on_training_iteration(self, *, algorithm, result, **kwargs):
        """Called at the end of each training iteration."""
        super().on_training_iteration(algorithm=algorithm, result=result, **kwargs)
        
        # `result["timesteps_total"]` is how many env steps have been sampled overall
        # or we can use "training_iteration" if we want iteration-based scheduling.
        current_timestep = result.get("timesteps_total", 0)
        
        # Fraction of the way through the schedule
        fraction = min(float(current_timestep) / self.anneal_timesteps, 1.0)
        
        # Linear annealing from initial_beta to final_beta
        new_beta = self.initial_beta + fraction * (self.final_beta - self.initial_beta)
        
        # Access the "local replay buffer" in RLlib and set its beta
        # With MultiAgentPrioritizedReplayBuffer, there is a `replay_buffers` dict for each policy
        if hasattr(algorithm, "local_replay_buffer") and hasattr(algorithm.local_replay_buffer, "replay_buffers"):
            for buf in algorithm.local_replay_buffer.replay_buffers.values():
                buf.beta = new_beta
        elif hasattr(algorithm, "local_replay_buffer"):
            # Single-agent or if the above is not needed
            algorithm.local_replay_buffer.beta = new_beta
        
        # OPTIONAL: add custom metrics so you can see beta in logs
        result["custom_metrics"]["current_beta"] = new_beta
        if fraction >= 1.0:
            result["custom_metrics"]["beta_status"] = "fully_annealed"