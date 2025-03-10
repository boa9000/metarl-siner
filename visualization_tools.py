import matplotlib.pyplot as plt

class GraphSaver:
    def __init__(self, plot_file, meta_plot_file, test_ids, env_ids):
        self.PLOT_FILE = plot_file
        self.META_PLOT_FILE = meta_plot_file
        self.test_ids = test_ids
        self.env_ids = env_ids

    def save_graph(self, avg_rewards, iteration):
        fig = plt.figure()
        plt.subplot(111)
        plt.ylabel("avg_Reward")
        plt.plot(avg_rewards)
        fig.savefig(self.PLOT_FILE)
        plt.close(fig)

    def save_meta_graph(self, rewards_with_model, rewards_without_model, rewards_dummy):
        num_envs = len(self.test_ids)
        fig_width = 3.5  # Single-column width in inches
        fig_height = 2.5 * num_envs  # Scale height based on number of subplots
        fig, axs = plt.subplots(num_envs, 1, figsize=(fig_width, fig_height), dpi=300)

        if num_envs == 1:
            axs = [axs]

        for i, env in enumerate(self.test_ids):
            axs[i].plot(rewards_with_model[env], label='With Meta Model')
            axs[i].plot(rewards_without_model[env], label='Without Meta Model')
            axs[i].plot(rewards_dummy[env], label='No Actions')
            axs[i].set_title(f'Rewards for {env}', fontsize=8)  # Reduce font size for IEEE compatibility
            axs[i].set_ylabel('Reward', fontsize=8)
            axs[i].legend(fontsize=6)
            axs[i].tick_params(axis='both', which='major', labelsize=6)

        plt.xlabel("Episode", fontsize=8)
        plt.tight_layout()
        fig.savefig(self.META_PLOT_FILE, dpi = 300, bbox_inches="tight")
        plt.close(fig)

    def save_learning_curves(self, rewards, iteration):
        fig, axs = plt.subplots(len(self.env_ids), 1, figsize=(10, 5 * len(self.env_ids)))
        for i, env in enumerate(self.env_ids):
            axs[i].plot(rewards[env], label=f'Env {env}')
            axs[i].set_title(f'Environment: {env}')
            axs[i].set_ylabel('Reward')
            axs[i].set_xlabel('Episode')
            axs[i].legend()
        plt.tight_layout()
        fig.savefig(f"{self.PLOT_FILE[:-4]}_learning_curves_iter_{iteration}.png")
        plt.close(fig)

    def save_validation_curve(self, vali_curve, iteration, approach):
        fig_width = 3.5 
        fig_height = 2.5 

        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300) 

        ax.plot(vali_curve, linewidth=1)
        ax.set_ylabel("Difference (Meta vs. Metaless)", fontsize=8)
        ax.set_xlabel("Validation iteration", fontsize=8)
        ax.set_title("Validation Curve", fontsize=9)
        
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

        plt.tight_layout(pad=0.3)
        fig.savefig(f"{self.PLOT_FILE[:-4]}_validation_curve_{approach}.png", dpi=300, bbox_inches="tight")
        plt.close(fig) 

    @staticmethod
    def relative_improvement(rewards):
        initial_reward = rewards[0]
        final_reward = rewards[-1]
        return (final_reward - initial_reward) / (abs(initial_reward) + 1e-8)  # Avoid division by zero