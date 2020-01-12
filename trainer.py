import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import SkipGramEmbeddings
from sgns_loss import SGNSLoss
from tqdm import tqdm
from datasets.pypi_lang import PyPILangDataset
from datasets.world_order import WorldOrderDataset
from torch.utils.tensorboard import SummaryWriter


class Trainer:

    def __init__(self, args):
        # Load data
        self.args = args
        self.writer = SummaryWriter(log_dir='./experiments/', flush_secs=3)
        #self.dataset = PyPILangDataset(args, examples_path='data/pypi_examples.pth', dict_path='data/pypi_dictionary.gensim')
        self.dataset = PyPILangDataset(args)
        self.vocab_size = len(self.dataset.dictionary)
        print("Finished loading dataset")

        self.dataloader = DataLoader(self.dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers)

        self.model = SkipGramEmbeddings(self.vocab_size, args.embedding_len).to(args.device)
        self.optim = optim.SparseAdam(self.model.parameters(), lr=args.lr)
        self.sgns = SGNSLoss(self.dataset, self.model.word_embeds, self.model.context_embeds, self.args.device)

        # Add graph to tensorboard
        #self.writer.add_graph(self.model, iter(self.dataloader).next()[0])

    def train(self):
        print('Training on device: {}'.format(self.args.device))

        # Log embeddings!
        print('\nRandom embeddings:')
        for word in self.dataset.queries:
            print(f'word: {word} neighbors: {self.model.nearest_neighbors(word, self.dataset.dictionary)}')

        for epoch in range(self.args.epochs):

            print(f'Beginning epoch: {epoch + 1}/{self.args.epochs}')
            running_loss = 0.0
            global_step = epoch * len(self.dataloader)
            num_examples = 0

            for i, data in enumerate(tqdm(self.dataloader)):
                # Unpack data
                center, context = data
                center, context = center.to(self.args.device), context.to(self.args.device)

                # Remove accumulated gradients
                self.optim.zero_grad()

                # Get context vector: word + doc
                center_embed, context_embed = self.model(center, context)

                # Calc loss: SGNS
                loss = self.sgns(center_embed, context_embed)

                # Backprop and update
                loss.backward()
                self.optim.step()

                # Keep track of loss
                running_loss += loss.item()
                global_step += 1
                num_examples += len(data)  # Last batch's size may not equal args.batch_size

                # Log at step
                #if global_step % self.args.log_step == 0:
                #    norm = (i + 1) * num_examples
                #    self.log_step(epoch, global_step, running_loss/norm, center, context)

            norm = (i + 1) * num_examples
            self.log_and_save_epoch(epoch, running_loss / norm)
            self.log_step(epoch, global_step, running_loss / norm, center, context)

        self.writer.close()

    def log_and_save_epoch(self, epoch, loss):
        # Visualize document embeddings
        self.writer.add_embedding(
            self.model.word_embeds.weight,
            global_step=epoch,
            tag=f'we_epoch_{epoch}',
        )

        # Save checkpoint
        print(f'Beginning to save checkpoint')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
            'loss': loss,
        }, f'epoch_{epoch}_ckpt.pth')
        print(f'Finished saving checkpoint')

    def log_step(self, epoch, global_step, loss, center, target):
        print(f'#############################################')
        print(f'EPOCH: {epoch} | STEP: {global_step} | LOSS {loss}')
        print(f'#############################################')

        self.writer.add_scalar('train_loss', loss, global_step)

        # Log embeddings!
        print('\nLearned embeddings:')
        for word in self.dataset.queries:
            print(f'word: {word} neighbors: {self.model.nearest_neighbors(word, self.dataset.dictionary)}')
