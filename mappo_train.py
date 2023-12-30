import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse
from normalization import Normalization, RewardScaling
from replay_buffer import ReplayBuffer
from mappo_agent import MAPPO_MPE
from env import Env


class Runner_MAPPO_MPE:
    def __init__(self, args, env_name, number, seed):
        self.args = args
        self.env_name = env_name
        self.number = number
        self.seed = seed
        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # Create env
        self.env = Env(args)
        self.args.N = self.env.n  # The number of agents
        self.args.obs_dim_n = [self.env.observation_space for i in range(self.args.N)]  # obs dimensions of N agents
        self.args.action_dim_n = [self.env.action_space for i in range(self.args.N)]  # actions dimensions of N agents
        # Only for homogenous agents environments like Spread in MPE,all agents have the same dimension of observation space and action space
        self.args.obs_dim = self.args.obs_dim_n[0]  # The dimensions of an agent's observation space
        self.args.action_dim = self.args.action_dim_n[0]  # The dimensions of an agent's action space
        self.args.state_dim = np.sum(self.args.obs_dim_n)  # The dimensions of global state space（Sum of the dimensions of the local observation space of all agents）
        print("observation_space=", self.env.observation_space)
        print("obs_dim_n={}".format(self.args.obs_dim_n))
        print("action_space=", self.env.action_space)
        print("action_dim_n={}".format(self.args.action_dim_n))

        # Create N agents
        self.agent_n = MAPPO_MPE(self.args)
        self.replay_buffer = ReplayBuffer(self.args)

        # Create a tensorboard
        self.writer = SummaryWriter(log_dir='runs/MAPPO/MAPPO_env_{}_number_{}_seed_{}'.format(self.env_name, self.number, self.seed))

        self.evaluate_rewards = []  # Record the rewards during the evaluating
        self.total_steps = 0
        if self.args.use_reward_norm:
            print("------use reward norm------")
            self.reward_norm = Normalization(shape=self.args.N)
        elif self.args.use_reward_scaling:
            print("------use reward scaling------")
            self.reward_scaling = RewardScaling(shape=self.args.N, gamma=self.args.gamma)

    def run(self, ):
        evaluate_num = -1  # Record the number of evaluations
        while self.total_steps < self.args.max_train_steps:
            if self.total_steps // self.args.evaluate_freq > evaluate_num:
                self.evaluate_policy()  # Evaluate the policy every 'evaluate_freq' steps
                evaluate_num += 1

            _, episode_steps = self.run_episode_mpe(evaluate=False)  # Run an episode

            self.total_steps += episode_steps
            # print("over one episode,total_step:", self.total_steps)
            if self.replay_buffer.episode_num == self.args.batch_size:
                self.agent_n.train(self.replay_buffer, self.total_steps)  # Training
                self.replay_buffer.reset_buffer()

        self.evaluate_policy()
        self.env.close()

    def evaluate_policy(self, ):
        evaluate_reward = 0
        evaluate_accept_order = 0
        evaluate_overdue_order = 0
        evaluate_data_collected = 0
        evaluate_ava_AoI = 0
        evaluate_data_q = 0
        for _ in range(self.args.evaluate_times):
            episode_reward, _, accept_order, overdue_order, data_collected, ava_AoI, data_q = self.run_episode_mpe(evaluate=True)
            evaluate_reward += episode_reward
            evaluate_accept_order += accept_order
            evaluate_overdue_order += overdue_order
            evaluate_data_collected += data_collected
            evaluate_ava_AoI += ava_AoI
            evaluate_data_q += data_q

        evaluate_reward = evaluate_reward / self.args.evaluate_times
        evaluate_accept_order = evaluate_accept_order / self.args.evaluate_times
        evaluate_overdue_order = evaluate_overdue_order / self.args.evaluate_times
        evaluate_data_collected = evaluate_data_collected / self.args.evaluate_times
        evaluate_ava_AoI = evaluate_ava_AoI / self.args.evaluate_times
        evaluate_data_q = evaluate_data_q / self.args.evaluate_times
        self.evaluate_rewards.append(evaluate_reward)
        print("total_steps:{} \t evaluate_reward:{}".format(self.total_steps, evaluate_reward),
              " evaluate_accept_order:{}".format(evaluate_accept_order),
              " evaluate_overdue_order:{}".format(evaluate_overdue_order),
              " evaluate_data_collected:{}".format(evaluate_data_collected),
              " evaluate_ava_AoI:{}".format(evaluate_ava_AoI),
              " evaluate_data_q:{}".format(evaluate_data_q))

        self.writer.add_scalar('evaluate_step_rewards_{}'.format(self.env_name), evaluate_reward, global_step=self.total_steps)
        # Save the rewards and models
        self.writer.add_scalar('evaluate_accept_order_{}'.format(self.env_name), evaluate_accept_order, global_step=self.total_steps)
        self.writer.add_scalar('evaluate_overdue_order_{}'.format(self.env_name), evaluate_overdue_order, global_step=self.total_steps)
        self.writer.add_scalar('evaluate_data_collected_{}'.format(self.env_name), evaluate_data_collected, global_step=self.total_steps)
        self.writer.add_scalar('evaluate_data_q_{}'.format(self.env_name), evaluate_data_q, global_step=self.total_steps)
        np.save('./data_train/MAPPO_env_{}_number_{}_seed_{}.npy'.format(self.env_name, self.number, self.seed), np.array(self.evaluate_rewards))
        self.agent_n.save_model(self.env_name, self.number, self.seed, self.total_steps)

    def run_episode_mpe(self, evaluate=False):
        episode_reward = 0
        obs_n = self.env.reset()
        if self.args.use_reward_scaling:
            self.reward_scaling.reset()

        for episode_step in range(self.args.episode_limit):
            a_n, a_logprob_n = self.agent_n.choose_action(obs_n, evaluate=evaluate)  # Get actions and the corresponding log probabilities of N agents
            s = np.array(obs_n).flatten()  # In MPE, global state is the concatenation of all agents' local obs.
            v_n = self.agent_n.get_value(s)  # Get the state values (V(s)) of N agents
            obs_next_n, r_n, done_n, _ = self.env.step(a_n)
            episode_reward += r_n[0]

            if not evaluate:
                if self.args.use_reward_norm:
                    r_n = self.reward_norm(r_n)
                elif args.use_reward_scaling:
                    r_n = self.reward_scaling(r_n)

                # Store the transition
                self.replay_buffer.store_transition(episode_step, obs_n, s, v_n, a_n, a_logprob_n, r_n, done_n)

            obs_n = obs_next_n
            if all(done_n):
                break

        if not evaluate:
            # An episode is over, store v_n in the last step
            s = np.array(obs_n).flatten()
            v_n = self.agent_n.get_value(s)
            self.replay_buffer.store_last_value(episode_step + 1, v_n)
            return episode_reward, episode_step + 1
        else:
            return episode_reward, episode_step + 1, self.env.accepted_order_number, self.env.overdue_order_number,\
                   self.env.data_collected, self.env.ava_AoI, self.env.data_q


if __name__ == '__main__':
    import dgl

    u = torch.load("./dataset/graph/35_region_u.pt")
    v = torch.load("./dataset/graph/35_region_v.pt")
    # u, v = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9]), \
    #        torch.tensor([1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2, 1, 2, 3, 0, 1, 3, 1, 3, 4, 4, 5, 6, 4, 6, 7, 5, 6, 7])
    region_graph = dgl.graph((u, v))

    parser = argparse.ArgumentParser("Hyperparameters Setting for MAPPO in MPE environment")
    parser.add_argument("--max_train_steps", type=int, default=int(5e6), help=" Maximum number of training steps")
    parser.add_argument("--episode_limit", type=int, default=200, help="Maximum number of steps per episode")
    parser.add_argument("--evaluate_freq", type=float, default=1600, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=40, help="Evaluate times")

    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (the number of episodes)")
    parser.add_argument("--mini_batch_size", type=int, default=8, help="Minibatch size (the number of episodes)")
    parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="The number of neurons in hidden layers of the rnn")
    parser.add_argument("--mlp_hidden_dim", type=int, default=64, help="The number of neurons in hidden layers of the mlp")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GAE parameter")
    parser.add_argument("--K_epochs", type=int, default=15, help="GAE parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=True, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling. Here, we do not use it.")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_relu", type=float, default=False, help="Whether to use relu, if False, we will use tanh")
    parser.add_argument("--use_gnn", type=bool, default=False, help="Whether to use GNN")
    parser.add_argument("--add_agent_id", type=float, default=False, help="Whether to add agent_id. Here, we do not use it.")
    parser.add_argument("--use_value_clip", type=float, default=False, help="Whether to use value clip.")

    parser.add_argument("--agent_n", type=int, default=100, help="The number of agent.")
    parser.add_argument("--region_n", type=int, default=35, help="The number of region.")
    parser.add_argument("--region_graph", default=region_graph, help="Region graph.")
    parser.add_argument("--order_data_path", default="D:/myCode/crowdSensing/dataset/5-10.csv", help="Dataset path.")
    parser.add_argument("--alpha", type=float, default=0.7, help="Reward weight alpha.")
    parser.add_argument("--beta", type=float, default=0.3, help="Reward weight beta.")
    parser.add_argument("--seed", type=float, default=20, help="Random seed.")

    args = parser.parse_args()
    runner = Runner_MAPPO_MPE(args, env_name="rb", number=2, seed=args.seed)
    runner.run()