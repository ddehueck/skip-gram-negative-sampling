import argparse
from trainer import Trainer
import torch as t


def get_args():
    parser = argparse.ArgumentParser(description="PyTorch LDA2Vec Training")

    """
    Data handling
    """
    parser.add_argument('--dataset-dir', type=str, default='data/',
                        help='dataset directory (default: data/)')
    parser.add_argument('--workers', type=int, default=4, metavar='N',
                       help='dataloader threads (default: 4)')
    parser.add_argument('--window-size', type=int, default=5, help='Window size\
                        used when generating training examples (default: 5)')
    parser.add_argument('--file-batch-size', type=int, default=250, help='Batch size\
                        used when multi-threading the generation of training examples\
                        (default: 250)')

    """
    Model Parameters
    """
    parser.add_argument('--embedding-len', type=int, default=128, help='Length of\
                        embeddings in model (default: 128)')

    """
    Training Hyperparameters
    """
    parser.add_argument('--epochs', type=int, default=15, metavar='N',
                        help='number of epochs to train for - iterations over the dataset (default: 15)')
    parser.add_argument('--batch-size', type=int, default=1024,
                        metavar='N', help='number of examples in a training batch (default: 1024)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')

    """
    Checkpoint options
    """
    parser.add_argument('--log-step', type=int, default=250, help='Step at which for every step training info\
                        is logged. (default: 250)')

    """
    Training Settings
    """
    parser.add_argument('--device', type=str, default=t.device("cuda:0" if t.cuda.is_available() else "cpu"),
                        help='device to train on (default: cuda:0 if cuda is available otherwise cpu)')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    trainer = Trainer(args)
    # Begin Training!
    trainer.train()