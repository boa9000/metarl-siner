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
        fig, axs = plt.subplots(num_envs, 1, figsize=(12, 6 * num_envs))

        if num_envs == 1:
            axs = [axs]

        for i, env in enumerate(self.test_ids):
            axs[i].plot(rewards_with_model[env], label='With Meta Model')
            axs[i].plot(rewards_without_model[env], label='Without Meta Model')
            axs[i].plot(rewards_dummy[env], label='No Actions')
            axs[i].set_title(f'Rewards for {env}')
            axs[i].set_ylabel('Reward')
            axs[i].legend()

        plt.tight_layout()
        fig.savefig(self.META_PLOT_FILE)
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
        fig.savefig(f"{self.PLOT_FILE}_learning_curves_iter_{iteration}.png")
        plt.close(fig)

    def save_validation_curve(self, vali_curve, iteration):
        fig = plt.figure()
        plt.plot(vali_curve)
        plt.ylabel("Difference between Meta and Metaless")
        plt.xlabel("Episode")
        plt.title("Validation curve over episodes")
        fig.savefig(f"{self.PLOT_FILE}_validation_curve.png")
        plt.close

    @staticmethod
    def relative_improvement(rewards):
        initial_reward = rewards[0]
        final_reward = rewards[-1]
        return (final_reward - initial_reward) / (abs(initial_reward) + 1e-8)  # Avoid division by zero