def custom_metrics_fn(episodes):
    return {
        "is_success": [
            bool(episode.last_info_for().get("is_success", False))
            for episode in episodes
        ]
    }