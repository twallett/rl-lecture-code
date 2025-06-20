import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def violinplot_environment(data, arm_means=None):
    """
    Create violin plots for multi-armed bandit reward distributions.
    
    Args:
        data: numpy array of shape (n_samples, n_arms) containing reward samples
        arm_means: list of arrays containing true means for different environments (optional)
    """
    legend_labels = []
    
    if arm_means is None:
        plt.figure(figsize=(10, 6))
        parts = plt.violinplot(data, showmeans=True)
        plt.xticks(range(1, data.shape[1] + 1))
        legend_labels.append(f"Distribution ({data.shape[0]} samples)")
        plt.xlabel("Arms (Possible actions)")
        plt.ylabel("Reward distribution")
        plt.title(f"Multi-Armed Bandit - {data.shape[1]} Arms Reward Distribution")
        plt.legend(labels=legend_labels, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.show()
    else:
        plt.figure(figsize=(12, 6))
        n_environments = len(arm_means)
        n_arms = data.shape[1]
        samples_per_env = data.shape[0] // n_environments
        
        legend_handles = []
        positions = []
    
        arm_width = 0.8
        env_spacing = arm_width / n_environments
        
        for arm_idx in range(n_arms):
            arm_center = arm_idx + 1
            for env_idx in range(n_environments):
                offset = (env_idx - (n_environments - 1) / 2) * env_spacing
                positions.append(arm_center + offset)
        
        for env_idx in range(n_environments):
            start_idx = env_idx * samples_per_env
            end_idx = (env_idx + 1) * samples_per_env
            color = f'C{env_idx}'
            
            for arm_idx in range(n_arms):
                pos_idx = env_idx * n_arms + arm_idx
                pos = positions[pos_idx]
                
                parts = plt.violinplot(data[start_idx:end_idx, arm_idx], 
                                     positions=[pos], showmeans=True, widths=env_spacing*0.8)
                
                parts['bodies'][0].set_color(color)
                parts['bodies'][0].set_alpha(0.7)
                parts['cmeans'].set_color(color)
                parts['cbars'].set_color(color)
                parts['cmins'].set_color(color)
                parts['cmaxes'].set_color(color)
            
            legend_labels.append(f"Environment {env_idx + 1} (samples {start_idx}-{end_idx-1})")
            legend_handles.append(Rectangle((0, 0), 1, 1, color=color, alpha=0.7))
        
        plt.xticks(range(1, n_arms + 1), [f"Arm {i+1}" for i in range(n_arms)])
        plt.xlabel("Arms (Possible actions)")
        plt.ylabel("Reward distribution")
        plt.title(f"Multi-Armed Bandit - {n_arms} Arms, {n_environments} Environments")
        plt.legend(handles=legend_handles, labels=legend_labels, 
                  loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.show()

def data_average_plot(data, arm_means, top=None):
    """
    Plot rolling average of rewards for each arm with true means.
    
    Args:
        data: numpy array of shape (n_samples, n_arms) containing reward samples
        arm_means: list of arrays containing true means for different environments
        top: int, number of top arms to display (optional)
    """
    plot_data = data.copy()
    plot_arm_means = arm_means.copy() if arm_means else None
    arm_indices = np.arange(data.shape[1])
    
    if top is not None and top < data.shape[1]:
        if arm_means:
            mean_rewards = np.mean([np.array(means) for means in arm_means], axis=0)
            top_indices = np.argsort(mean_rewards)[-top:]
            plot_data = data[:, top_indices]
            plot_arm_means = [np.array(means)[top_indices] for means in arm_means]
            arm_indices = top_indices
        else:
            empirical_means = np.mean(data, axis=0)
            top_indices = np.argsort(empirical_means)[-top:]
            plot_data = data[:, top_indices]
            arm_indices = top_indices
    
    plt.figure(figsize=(12, 6))
    
    for j in range(plot_data.shape[1]):
        cumsum = np.cumsum(plot_data[:, j])
        rolling_avg = cumsum / np.arange(1, len(cumsum) + 1)
        plt.plot(rolling_avg, color=f"C{j}", label=f"Arm {arm_indices[j] + 1}", linewidth=2)
    
    if plot_arm_means:
        if len(plot_arm_means) == 1:
            for j in range(plot_data.shape[1]):
                true_mean = plot_arm_means[0][j]
                plt.axhline(y=true_mean, color=f"C{j}", linestyle="--", alpha=0.7)
        else:
            samples_per_env = plot_data.shape[0] // len(plot_arm_means)
            for env_idx, means in enumerate(plot_arm_means):
                start_x = env_idx * samples_per_env
                end_x = (env_idx + 1) * samples_per_env
                for j in range(plot_data.shape[1]):
                    plt.plot([start_x, end_x], [means[j], means[j]], 
                           color=f"C{j}", linestyle="--", alpha=0.7, linewidth=2)
    
    plt.xlabel("Time Steps")
    plt.ylabel("Average Reward")
    plt.title("Multi-Armed Bandit - Rolling Average Rewards")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def data_cumulative_plot(data, arm_means, top=None):
    """
    Plot cumulative rewards for each arm with expected cumulative rewards.
    
    Args:
        data: numpy array of shape (n_samples, n_arms) containing reward samples
        arm_means: list of arrays containing true means for different environments
        top: int, number of top arms to display (optional)
    """
    plot_data = data.copy()
    plot_arm_means = arm_means.copy() if arm_means else None
    arm_indices = np.arange(data.shape[1])
    
    if top is not None and top < data.shape[1]:
        if arm_means:
            mean_rewards = np.mean([np.array(means) for means in arm_means], axis=0)
            top_indices = np.argsort(mean_rewards)[-top:]
            plot_data = data[:, top_indices]
            plot_arm_means = [np.array(means)[top_indices] for means in arm_means]
            arm_indices = top_indices
        else:
            empirical_means = np.mean(data, axis=0)
            top_indices = np.argsort(empirical_means)[-top:]
            plot_data = data[:, top_indices]
            arm_indices = top_indices
    
    plt.figure(figsize=(12, 6))
    
    for j in range(plot_data.shape[1]):
        cumsum = np.cumsum(plot_data[:, j])
        plt.plot(cumsum, color=f"C{j}", label=f"Arm {arm_indices[j] + 1}", linewidth=2)
    
    if plot_arm_means:
        time_steps = np.arange(1, plot_data.shape[0] + 1)
        
        if len(plot_arm_means) == 1:
            for j in range(plot_data.shape[1]):
                expected_cumsum = plot_arm_means[0][j] * time_steps
                plt.plot(time_steps, expected_cumsum, color=f"C{j}", 
                        linestyle="--", alpha=0.7, linewidth=2)
        else:
            samples_per_env = plot_data.shape[0] // len(plot_arm_means)
            for j in range(plot_data.shape[1]):
                expected_cumsum = np.zeros(plot_data.shape[0])
                for env_idx, means in enumerate(plot_arm_means):
                    start_idx = env_idx * samples_per_env
                    end_idx = (env_idx + 1) * samples_per_env
                    
                    if env_idx == 0:
                        expected_cumsum[start_idx:end_idx] = means[j] * np.arange(1, samples_per_env + 1)
                    else:
                        prev_cumsum = expected_cumsum[start_idx - 1]
                        expected_cumsum[start_idx:end_idx] = (prev_cumsum + 
                                                            means[j] * np.arange(1, samples_per_env + 1))
                
                plt.plot(time_steps, expected_cumsum, color=f"C{j}", 
                        linestyle="--", alpha=0.7, linewidth=2)
    
    plt.xlabel("Time Steps")
    plt.ylabel("Cumulative Reward")
    plt.title("Multi-Armed Bandit - Cumulative Rewards")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def model_average_plot(data, rewards, action_matrix, arm_means, top=None):
    """
    Plot model performance: rolling average of actual rewards vs arm selection frequency.
    
    Args:
        data: numpy array of environment data (not used directly, kept for compatibility)
        rewards: 1D array of actual rewards received by the agent
        action_matrix: binary matrix of shape (n_steps, n_arms) indicating arm selections
        arm_means: list of true arm means
        top: int, number of top arms to display (optional)
    """
    plot_action_matrix = action_matrix.copy()
    plot_arm_means = arm_means.copy() if arm_means else None
    arm_indices = np.arange(action_matrix.shape[1])
    
    if top is not None and top < action_matrix.shape[1]:
        if arm_means:
            mean_rewards = np.mean([np.array(means) for means in arm_means], axis=0)
            top_indices = np.argsort(mean_rewards)[-top:]
            plot_action_matrix = action_matrix[:, top_indices]
            plot_arm_means = [np.array(means)[top_indices] for means in arm_means]
            arm_indices = top_indices
    
    plt.figure(figsize=(12, 6))
    
    cumsum_rewards = np.cumsum(rewards)
    rolling_avg_rewards = cumsum_rewards / np.arange(1, len(rewards) + 1)
    plt.plot(rolling_avg_rewards, color="black", linewidth=3, label="Actual Rewards")
    
    for j in range(plot_action_matrix.shape[1]):
        cumsum_selections = np.cumsum(plot_action_matrix[:, j])
        selection_freq = cumsum_selections / np.maximum(np.arange(1, len(cumsum_selections) + 1), 1)
        plt.plot(selection_freq, color=f"C{j}", linewidth=2, 
                label=f"Arm {arm_indices[j] + 1} Selection Rate")
    
    if plot_arm_means:
        if len(plot_arm_means) == 1:
            for j in range(plot_action_matrix.shape[1]):
                plt.axhline(y=plot_arm_means[0][j], color=f"C{j}", 
                          linestyle="--", alpha=0.7)
        else:
            samples_per_env = len(rewards) // len(plot_arm_means)
            for env_idx, means in enumerate(plot_arm_means):
                start_x = env_idx * samples_per_env
                end_x = (env_idx + 1) * samples_per_env
                for j in range(plot_action_matrix.shape[1]):
                    plt.plot([start_x, end_x], [means[j], means[j]], 
                           color=f"C{j}", linestyle="--", alpha=0.7, linewidth=2)
    
    plt.xlabel("Time Steps")
    plt.ylabel("Average Reward / Selection Rate")
    plt.title("Multi-Armed Bandit - Model Performance")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def model_cumulative_plot(data, rewards, action_matrix, arm_means, top=None):
    """
    Plot model performance: cumulative rewards vs expected cumulative rewards.
    
    Args:
        data: numpy array of environment data (not used directly)
        rewards: 1D array of actual rewards received by the agent
        action_matrix: binary matrix indicating arm selections
        arm_means: list of true arm means
        top: int, number of top arms to display (optional)
    """
    plot_action_matrix = action_matrix.copy()
    plot_arm_means = arm_means.copy() if arm_means else None
    arm_indices = np.arange(action_matrix.shape[1])
    
    if top is not None and top < action_matrix.shape[1]:
        if arm_means:
            mean_rewards = np.mean([np.array(means) for means in arm_means], axis=0)
            top_indices = np.argsort(mean_rewards)[-top:]
            plot_action_matrix = action_matrix[:, top_indices]
            plot_arm_means = [np.array(means)[top_indices] for means in arm_means]
            arm_indices = top_indices
    
    plt.figure(figsize=(12, 6))
    
    cumsum_rewards = np.cumsum(rewards)
    plt.plot(cumsum_rewards, color="black", linewidth=3, label="Actual Cumulative Rewards")
    
    for j in range(plot_action_matrix.shape[1]):
        cumsum_selections = np.cumsum(plot_action_matrix[:, j])
        plt.plot(cumsum_selections, color=f"C{j}", linewidth=2, 
                label=f"Arm {arm_indices[j] + 1} Selections")
    
    if plot_arm_means:
        time_steps = np.arange(1, len(rewards) + 1)
        
        if len(plot_arm_means) == 1:
            for j in range(plot_action_matrix.shape[1]):
                expected_cumsum = plot_arm_means[0][j] * time_steps
                plt.plot(time_steps, expected_cumsum, color=f"C{j}", 
                        linestyle="--", alpha=0.7, linewidth=2,
                        label=f"Expected Arm {arm_indices[j] + 1}")
        else:
            samples_per_env = len(rewards) // len(plot_arm_means)
            for j in range(plot_action_matrix.shape[1]):
                expected_cumsum = np.zeros(len(rewards))
                for env_idx, means in enumerate(plot_arm_means):
                    start_idx = env_idx * samples_per_env
                    end_idx = (env_idx + 1) * samples_per_env
                    
                    if env_idx == 0:
                        expected_cumsum[start_idx:end_idx] = means[j] * np.arange(1, samples_per_env + 1)
                    else:
                        prev_cumsum = expected_cumsum[start_idx - 1]
                        expected_cumsum[start_idx:end_idx] = (prev_cumsum + 
                                                            means[j] * np.arange(1, samples_per_env + 1))
                
                plt.plot(time_steps, expected_cumsum, color=f"C{j}", 
                        linestyle="--", alpha=0.7, linewidth=2)
    
    plt.xlabel("Time Steps")
    plt.ylabel("Cumulative Rewards / Selections")
    plt.title("Multi-Armed Bandit - Model Cumulative Performance")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()