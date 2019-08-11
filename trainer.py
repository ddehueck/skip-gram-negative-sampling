import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import SkipGramEmbeddings
from sgns_loss import SGNSLoss
from tqdm import tqdm
from datasets.opinrank import OpinRankDataset
from torch.utils.tensorboard import SummaryWriter


class Trainer:

    def __init__(self, args):
        # Load data
        self.args = args
        self.writer = SummaryWriter(log_dir='./experiments/', flush_secs=3)
        self.dataset = OpinRankDataset(args)
        print("Finished loading dataset")

        print('Saving Dataset...')
        torch.save({
             'examples': self.dataset.examples,
             'term_freq_dict': self.dataset.term_freq_dict,
         }, 'dataset.pth')
        print('Saved Dataset!')

        self.dataloader = DataLoader(self.dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.num_workers)

        self.model = SkipGramEmbeddings(len(self.dataset.term_freq_dict), args.embedding_len).to(args.device)
        self.optim = optim.SGD(self.model.parameters(), lr=args.lr)
        self.sgns = SGNSLoss(self.dataset, self.model.word_embeds, self.args.device)

        # Visualize RANDOM  embeddings
        print('Adding random embeddings')
        self.writer.add_embedding(
            self.model.word_embeds,
            global_step=-1,
            tag=f'random_embeds',
        )
        print('Finished adding random embeddings!')

        # Add graph to tensorboard
        # TODO: Get working on multi-gpu stuff
        self.writer.add_graph(self.model, iter(self.dataloader).next()[0])

    def train(self):
        print('Training on device: {}'.format(self.args.device))

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

                # Calc loss: SGNS + Dirichlet
                loss = self.sgns(center_embed, context_embed)

                # Backprop and update
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
                self.optim.step()

                # Keep track of loss
                running_loss += loss.item()
                global_step += 1
                num_examples += len(data)  # Last batch size may not equal args.batch_size

                # Log at step
                if global_step % self.args.log_step == 0:
                    norm = (i + 1) * num_examples
                    self.log_step(epoch, global_step, running_loss/norm, center, context)

            norm = (i + 1) * num_examples
            self.log_and_save_epoch(epoch, running_loss / norm)

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
        print(f'#############################################\n\n')

        self.writer.add_scalar('train_loss', loss, global_step)
        # Log gradients - index select to only view gradients of embeddings in batch
        print(f'WORD EMBEDDING GRADIENTS:\n\
            {torch.index_select(self.model.word_embeds.weight.grad, 0, center.squeeze())}')
        print(f'\n{torch.index_select(self.model.word_embeds.weight.grad, 0, target.squeeze())}')