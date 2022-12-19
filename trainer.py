import time
import random

import torch
import torch.nn as nn
import torch.optim as optim

from utils import epoch_time
from model.optim import ScheduledAdam
from model.transformer import Transformer
from tqdm import tqdm
import pickle
from bleu import get_bleu

random.seed(32)
torch.manual_seed(32)
torch.backends.cudnn.deterministic = True


class Trainer:
    def __init__(self, params, mode, train_iter=None, valid_iter=None, test_iter=None):
        self.params = params
        # Train mode
        if mode == 'train':
            self.train_iter = train_iter
            self.valid_iter = valid_iter

        # Test mode
        else:
            self.test_iter = test_iter

        self.model = Transformer(self.params)
        self.model.to(self.params.device)

        # Scheduling Optimzer
        self.optimizer = ScheduledAdam(
            optim.Adam(self.model.parameters(), betas=(0.9, 0.98), eps=1e-9),
            hidden_dim=params.hidden_dim,
            warm_steps=params.warm_steps
        )

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.params.pad_idx)
        self.criterion.to(self.params.device)

    def train(self):
        print(self.model)
        print(f'The model has {self.model.count_params():,} trainable parameters')
        best_valid_loss = float('inf')

        for epoch in range(self.params.num_epoch):
            self.model.train()
            epoch_loss = 0
            start_time = time.time()

            for i,batch in tqdm(enumerate(self.train_iter)):
                # For each batch, first zero the gradients
                self.optimizer.zero_grad()
                source = batch.source
                target = batch.target
                # target sentence consists of <sos> and following tokens (except the <eos> token)
                output = self.model(source, target[:, :-1])[0]

                # ground truth sentence consists of tokens and <eos> token (except the <sos> token)
                output = output.contiguous().view(-1, output.shape[-1])
                target = target[:, 1:].contiguous().view(-1)
                # output = [(batch size * target length - 1), output dim]
                # target = [(batch size * target length - 1)]
                loss = self.criterion(output, target)
                loss.backward()

                # clip the gradients to prevent the model from exploding gradient
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip)

                self.optimizer.step()

                # 'item' method is used to extract a scalar from a tensor which only contains a single value.
                epoch_loss += loss.item()

            train_loss = epoch_loss / len(self.train_iter)
            valid_loss,val_bleu = self.evaluate()

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), self.params.save_model)

            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f} | Val. Bleu: {val_bleu:.3f}')

    def evaluate(self):
        self.model.eval()
        epoch_loss = 0
        batch_bleu = []
        with torch.no_grad():
            for batch in self.valid_iter:
                source = batch.source
                target = batch.target
                target_ = target
                output = self.model(source, target[:, :-1])[0]
                output_ = output.squeeze(0).max(dim=-1)[1]
                output = output.contiguous().view(-1, output.shape[-1])
                target = target[:, 1:].contiguous().view(-1)

                loss = self.criterion(output, target)

                epoch_loss += loss.item()

                total_bleu = []
                for j in range(self.params.batch_size):
                  try:
                        target_goal_token = [self.target.vocab.itos[token] for token in target_[j]]
                        target_goal = target_goal_token[target_goal_token.index('<sos>')+1:target_goal_token.index('<eos>')]
                        target_result = ' '.join(target_goal)
                        target_result = target_result.replace(' <unk>','')
                        translated_token = [self.target.vocab.itos[token] for token in output_[j]]
                        translation = translated_token[:translated_token.index('<eos>')]
                        translation = ' '.join(translation)
                        bleu = get_bleu(hypotheses=translation.split(), reference=target_result.split())
                        total_bleu.append(bleu)
                  except:
                    pass
                total_bleu = sum(total_bleu) / len(total_bleu)
                batch_bleu.append(total_bleu)

        batch_bleu = sum(batch_bleu) / len(batch_bleu)
        return epoch_loss / len(self.valid_iter),batch_bleu

    def inference(self):
        self.model.load_state_dict(torch.load(self.params.save_model))
        self.model.eval()
        epoch_loss = 0
        batch_bleu = []
        with torch.no_grad():
            for batch in self.test_iter:
                source = batch.source
                target = batch.target
                target_ = target
                output = self.model(source, target[:, :-1])[0]
                output_ = output.squeeze(0).max(dim=-1)[1]
                output = output.contiguous().view(-1, output.shape[-1])
                target = target[:, 1:].contiguous().view(-1)

                loss = self.criterion(output, target)

                epoch_loss += loss.item()

                total_bleu = []

                for j in range(self.params.batch_size):
                  try:
                        target_goal_token = [self.target.vocab.itos[token] for token in target_[j]]
                        target_goal = target_goal_token[target_goal_token.index('<sos>')+1:target_goal_token.index('<eos>')]
                        target_result = ' '.join(target_goal)
                        target_result = target_result.replace(' <unk>','')
                        translated_token = [self.target.vocab.itos[token] for token in output_[j]]
                        translation = translated_token[:translated_token.index('<eos>')]
                        translation = ' '.join(translation)
                        bleu = get_bleu(hypotheses=translation.split(), reference=target_result.split())
                        total_bleu.append(bleu)
                  except:
                    pass
                total_bleu = sum(total_bleu) / len(total_bleu)
                batch_bleu.append(total_bleu)

        batch_bleu = sum(batch_bleu) / len(batch_bleu)
        test_loss = epoch_loss / len(self.test_iter)
        print(f'Test Loss: {test_loss:.3f}')
        print(f'Test Bleu: {batch_bleu:.3f}')

