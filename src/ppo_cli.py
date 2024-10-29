

import importlib
import argparse

from ppo import PPO


def main():
    parser = argparse.ArgumentParser(description=
        """CLI tool for training a PPO agent. The number of global env steps is
        given by global_steps = num_updates * num_envs * steps_per_env and a
        larger value will increase how long the algorithm will take to complete 
        training.""")

    parser.add_argument("experiment_name", type=str, help="Str: Name for the experiment")
    parser.add_argument("env_module", type=str, help="Str: Name of module containing env init function 'train_fn'")
    parser.add_argument("num_updates", type=int, help="Int: Number of training iterations to perform")
    parser.add_argument("num_envs", type=int, help="Int: Number of processes in the vectorised environment")
    parser.add_argument("steps_per_env", type=int, help="Int: Number of steps to perform per env per update")
    parser.add_argument("num_epochs", type=int, help="Int: Number of epochs to train for per update")
    parser.add_argument("batch_size", type=int, help="Int: The minibatch size")
    parser.add_argument("critic_coef", type=float, help="Float: The critic loss coefficient")
    parser.add_argument("entropy_coef", type=float, help="Float: The entropy loss coefficient")
    parser.add_argument("clip_ratio", type=float, help="Float: The clipping ratio in the PPO clip objective")
    parser.add_argument("max_grad_norm", type=float, help="Float: The global gradient norm clipping value")
    parser.add_argument("learning_rate", type=float, help="Float: The learning rate for the Adam optimiser")
    parser.add_argument("discount_factor", type=float, help="Float: The discount factor")
    parser.add_argument("gae_factor", type=float, help="Float: Lambda in generalised advantage estimation")
    parser.add_argument("norm_adv", type=int, help="Bool: Set 1 to normalise the advantages batch-wise")
    parser.add_argument("clip_va_loss", type=int, help="Bool: Set 1 to clip the critic loss like the policy")
    parser.add_argument("conv_net", type=int, help="Bool: Set 1 to use the convolutional network")
    parser.add_argument("joint_network", type=int, help="Bool: Set 1 to use a joint policy and value network backbone")
    parser.add_argument("--use_gpu", default=False, type=int, help="Bool: Set 1 to use the gpu if Pytorch is properly setup to use CUDA (Default: 0)")
    parser.add_argument("--target_div", default=None, type=float, help="Float: The kl_div threshold which early stops the current update step if exceeded (Default: None)")
    parser.add_argument("--render_every", default=0, type=int, help="Int: Creates a gif of an episode every render_every global env steps (Default: 0)")
    parser.add_argument("--render_fps", default=0.0, type=float, help="Float: Frames per second of the episode renders (Default: 0.0)")
    parser.add_argument("--early_stop_reward", default=None, type=float, help="Float: The early stopping reward threshold (Default: None)")

    args = parser.parse_args()
    print()

    env_module = importlib.import_module(args.env_module)
    env_fn = getattr(env_module, "train_fn")

    agent = PPO(
            args.discount_factor,
            args.gae_factor,
            args.norm_adv,
            args.clip_va_loss,
            args.conv_net,
            args.joint_network,
            args.use_gpu,
            project_name=args.experiment_name
            )

    agent.train(
            env_fn,
            args.num_updates,
            args.num_envs,
            args.steps_per_env,
            args.num_epochs,
            args.batch_size,
            args.critic_coef,
            args.entropy_coef,
            args.clip_ratio,
            args.max_grad_norm,
            args.learning_rate,
            args.target_div,
            args.render_every,
            args.render_fps,
            args.early_stop_reward
            )

if __name__ == "__main__":
    main()

