from train import train


def main():
    """
    Main function.
    """

    train_config = {
        "batch_size": 32,
        "seq_len": 4,
        "num_epochs": 500,
        "learning_rate": 1e-3,
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.1,
    }

    train(**train_config)


if __name__ == "__main__":
    main()
