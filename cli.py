import argparse

import defaults
import dqn


def positive_float(flt):
    value = float(flt)
    if value < 0.0:
        msg = "{} must be greater than zero".format(value)
        raise argparse.ArgumentTypeError(msg)
    else:
        return value


def positive_int(i):
    value = int(i)
    if value < 1:
        msg = "{} must be greater than one".format(value)
        raise argparse.ArgumentTypeError(msg)
    else:
        return value


def zero_to_one(flt):
    value = float(flt)
    if value < 0.0 or value > 1.0:
        msg = "{} must be between zero and one".format(value)
        raise argparse.ArgumentTypeError(msg)
    else:
        return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Train a new agent")
    train_parser.add_argument("-e",
                              "--env",
                              default=defaults.env,
                              type=str,
                              choices=["CartPole-v1", "MountainCar-v0"],
                              help="The OpenAI Gym environment to train the agent on",
                              dest="env")
    train_parser.add_argument("-lr",
                              "--learning-rate",
                              default=defaults.learning_rate,
                              type=positive_float,
                              help="The learning rate for the agent's neural network",
                              dest="learning_rate")
    train_parser.add_argument("-bs",
                              "--batch-size",
                              default=defaults.batch_size,
                              type=positive_int,
                              help="The number of previous experiences trained on after each action",
                              dest="batch_size")
    train_parser.add_argument("-dd",
                              "--decay-delay",
                              default=defaults.decay_delay,
                              type=int,
                              help="The number of episodes to wait before decaying epsilon",
                              dest="decay_delay")
    train_parser.add_argument("-df",
                              "--discount-factor",
                              default=defaults.discount_factor,
                              type=zero_to_one,
                              help="The amount to discount future rewards by compared to present rewards",
                              dest="discount_factor")
    train_parser.add_argument("-ep",
                              "--episodes",
                              default=defaults.episodes,
                              type=positive_int,
                              help="The number of episodes to train the agent on",
                              dest="episodes")
    train_parser.add_argument("-en",
                              "--epsilon",
                              default=defaults.epsilon,
                              type=zero_to_one,
                              help="The probability that the agent will choose a random action",
                              dest="epsilon")
    train_parser.add_argument("-ed",
                              "--epsilon-decay",
                              default=defaults.epsilon_decay,
                              type=zero_to_one,
                              help="The amount to decay epsilon by after every action",
                              dest="epsilon_decay")
    train_parser.add_argument("-em",
                              "--epsilon-min",
                              default=defaults.epsilon_min,
                              type=zero_to_one,
                              help="The minimum value epsilon can take",
                              dest="epsilon_min")
    train_parser.add_argument("-ml",
                              "--memory-len",
                              default=defaults.memory_len,
                              type=positive_int,
                              help="The length of the agent's replay memory",
                              dest="memory_len")
    train_parser.add_argument("-ur",
                              "--update-rate",
                              default=defaults.update_rate,
                              type=positive_int,
                              help="The number of episodes between each update of the target network",
                              dest="update_rate")
    train_parser.add_argument("-o",
                              "--output",
                              default=defaults.model,
                              type=str,
                              help="The filename of the trained agent",
                              dest="model_filename")

    run_parser = subparsers.add_parser("run", help="Run an agent")
    run_parser.add_argument("-i",
                            "--input",
                            default=defaults.model,
                            type=str,
                            help="The name of the .h5 file containing the pre-trained agent",
                            dest="model_filename")
    run_parser.add_argument("-e",
                            "--env",
                            default=defaults.env,
                            type=str,
                            choices=["CartPole-v1", "MountainCar-v0"],
                            help="The OpenAI Gym environment to run the agent on",
                            dest="env")

    gif_parser = subparsers.add_parser("gif", help="Generate a GIF of an agent running")
    gif_parser.add_argument("-i",
                            "--input",
                            default=defaults.model,
                            type=str,
                            help="The name of the .h5 file containing the pre-trained agent",
                            dest="model_filename")
    gif_parser.add_argument("-e",
                            "--env",
                            default=defaults.env,
                            type=str,
                            choices=["CartPole-v1", "MountainCar-v0"],
                            help="The OpenAI Gym environment to run the agent on",
                            dest="env")
    gif_parser.add_argument("-o",
                            "--output",
                            default=defaults.filename,
                            type=str,
                            help="The file name of the gif",
                            dest="gif_filename")

    args = vars(parser.parse_args())
    command = args["command"]
    model_filename = args["model_filename"]
    args.pop("command")
    args.pop("model_filename")

    if command == "train":
        agent = dqn.Agent(**args)
        agent.train_model(model_filename)
    elif command == "run":
        agent = dqn.Agent(**args)
        agent.load_model(model_filename)
        agent.run()
    elif command == "gif":
        gif_filename = args["gif_filename"]
        args.pop("gif_filename")
        agent = dqn.Agent(**args)
        agent.load_model(model_filename)
        agent.generate_gif(gif_filename)
