import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import numpy as np

from PositionalEncoding import PositionalEncoding

class Transformer(nn.Module):

    """
    Model from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """
    # Constructor
    def __init__(
        self,
        num_tokens,
        dim_model,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        dropout_p,
        device,
    ):
        super().__init__()

        # INFO
        self.model_type = "Transformer"
        self.dim_model = dim_model

        # LAYERS
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=5000
        )
        self.embedding = nn.Embedding(num_tokens, dim_model)
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
        )
        self.out = nn.Linear(dim_model, num_tokens)

        #! parameters for training
        self.opt = torch.optim.SGD(self.parameters(), lr=0.01)
        self.loss_fn = nn.CrossEntropyLoss()
        self.device = device
        
    def forward(self, src, tgt, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)

        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        src = self.embedding(src) * math.sqrt(self.dim_model)
        tgt = self.embedding(tgt) * math.sqrt(self.dim_model)
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)
        
        # We could use the parameter batch_first=True, but our KDL version doesn't support it yet, so we permute
        # to obtain size (sequence length, batch_size, dim_model),
        src = src.permute(1,0,2)
        tgt = tgt.permute(1,0,2)

        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        transformer_out = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)
        out = self.out(transformer_out)
        
        return out
      
    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        
        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]
        
        return mask
    
    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return (matrix == pad_token)

    def train_loop(self, dataloader):
        """
        Method from "A detailed guide to Pytorch's nn.Transformer() module.", by
        Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
        """
        
        self.train()
        total_loss = 0
        
        for batch in dataloader:
            X, y = batch[:, 0], batch[:, 1]
            X, y = torch.tensor(X).to(self.device), torch.tensor(y).to(self.device)

            # Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
            y_input = y[:,:-1]
            y_expected = y[:,1:]
            
            # Get mask to mask out the next words
            sequence_length = y_input.size(1)
            tgt_mask = self.get_tgt_mask(sequence_length).to(self.device)

            # Standard training except we pass in y_input and tgt_mask
            pred = self(X, y_input, tgt_mask)

            # Permute pred to have batch size first again
            pred = pred.permute(1, 2, 0)      
            loss = self.loss_fn(pred, y_expected)

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        
            total_loss += loss.detach().item()
            
        return total_loss / len(dataloader)

    def validation_loop(self, dataloader):
        """
        Method from "A detailed guide to Pytorch's nn.Transformer() module.", by
        Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
        """
        
        self.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                X, y = batch[:, 0], batch[:, 1]
                X, y = torch.tensor(X, dtype=torch.long, device=self.device), torch.tensor(y, dtype=torch.long, device=self.device)

                # Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
                y_input = y[:,:-1]
                y_expected = y[:,1:]
                
                # Get mask to mask out the next words
                sequence_length = y_input.size(1)
                tgt_mask = self.get_tgt_mask(sequence_length).to(self.device)

                # Standard training except we pass in y_input and src_mask
                pred = self(X, y_input, tgt_mask)

                # Permute pred to have batch size first again
                pred = pred.permute(1, 2, 0)      
                loss = self.loss_fn(pred, y_expected)
                total_loss += loss.detach().item()
            
        return total_loss / len(dataloader)
    

    def fit(self, train_dataloader, val_dataloader, epochs):
        """
        Method from "A detailed guide to Pytorch's nn.Transformer() module.", by
        Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
        """
        
        # Used for plotting later on
        train_loss_list, validation_loss_list = [], []
        
        print("Training and validating model")
        for epoch in range(epochs):
            print("-"*25, f"Epoch {epoch + 1}","-"*25)
            
            train_loss = self.train_loop(train_dataloader)
            train_loss_list += [train_loss]
            
            validation_loss = self.validation_loop(val_dataloader)
            validation_loss_list += [validation_loss]
            
            print(f"Training loss: {train_loss:.4f}")
            print(f"Validation loss: {validation_loss:.4f}")
            print()
            
        return train_loss_list, validation_loss_list