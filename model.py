
import torch
import numpy as np
import pprint
import json
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import HfApi
from IPython.display import HTML
import pandas as pd
from functools import partial


DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}



class AutoEncoder(nn.Module):

    def __init__(self, d_hidden, version='new' ,d_mlp=2048, dtype=torch.float16, l1_coeff=3e-4):

        self.dtype = torch.float16
        self.d_hidden = d_hidden
        self.d_mlp = d_mlp
        self.l1_coeff = l1_coeff

        super().__init__()

        if version == 'new':
            self.encoder_bias = nn.Parameter(torch.zeros(self.d_mlp, dtype=self.dtype))
            self.encoder = nn.Linear(self.d_mlp, self.d_hidden)
            self.decoder = nn.Linear(self.d_hidden, self.d_mlp)
            self.activation = nn.ReLU()

            self.forward_func = self.forward_new

        elif version == 'original':
            self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_mlp, d_hidden, dtype=dtype)))
            self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_hidden, d_mlp, dtype=dtype)))
            self.b_enc = nn.Parameter(torch.zeros(d_hidden, dtype=dtype))
            self.b_dec = nn.Parameter(torch.zeros(d_mlp, dtype=dtype))

            self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
            self.forward_func = self.forward_original

        self.to("cuda")


    def forward(self, x):
        
        x = x - self.encoder_bias
        x = self.encoder(x)
        x = self.acts = self.activation(x)
        x = self.decoder(x)

        return self.forward_func(x)
    

    def forward_new(self, x):
        x = x - self.encoder_bias
        x = self.encoder(x)
        x = self.acts = self.activation(x)
        x = self.decoder(x)

        return x

    def forward_original(self, x):
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc + self.b_enc)
        x_reconstruct = acts @ self.W_dec + self.b_dec

        return x_reconstruct
    

    @staticmethod
    def calculate_loss(self, true_x, reconstructed_x):
    
        l1_loss = self.l1_coeff * (self.acts.float().abs().sum())
        l2_loss = (reconstructed_x.float() - true_x.float()).pow(2).sum(-1).mean(0)

        return l1_loss + l2_loss


    def save_checkpoint(self, path):
        print(f'Saving checkpoint {path}')
        torch.save({
            'dtype': self.dtype,
            'd_hidden': self.d_hidden,
            'd_mlp': self.d_mlp,
            'l1_coeff': self.l1_coeff,
            'model_state_dict': self.state_dict()
        }, path)

    @staticmethod
    def load_checkpoint(path) -> 'AutoEncoder':
        checkpoint = torch.load(path)
        model = AutoEncoder(
            dtype=checkpoint['dtype'],
            d_hidden=checkpoint['d_hidden'],
            d_mlp=checkpoint['d_mlp'],
            l1_coeff=checkpoint['l1_coeff']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    


def replacement_hook(mlp_post, hook, encoder):
    mlp_post_reconstr = encoder(mlp_post)[1]
    return mlp_post_reconstr

def mean_ablate_hook(mlp_post, hook):
    mlp_post[:] = mlp_post.mean([0, 1])
    return mlp_post

def zero_ablate_hook(mlp_post, hook):
    mlp_post[:] = 0.
    return mlp_post

@torch.no_grad()
def get_recons_loss(all_tokens, model, model_batch_size, num_batches=5, local_encoder=None):

    loss_list = []
    for i in range(num_batches):
        tokens = all_tokens[torch.randperm(len(all_tokens))[:model_batch_size]]
        loss = model(tokens, return_type="loss")
        recons_loss = model.run_with_hooks(tokens, return_type="loss", fwd_hooks=[(partial(replacement_hook, encoder=local_encoder))])
        # mean_abl_loss = model.run_with_hooks(tokens, return_type="loss", fwd_hooks=[(utils.get_act_name("post", 0), mean_ablate_hook)])
        zero_abl_loss = model.run_with_hooks(tokens, return_type="loss", fwd_hooks=[(zero_ablate_hook)])
        loss_list.append((loss, recons_loss, zero_abl_loss))
    losses = torch.tensor(loss_list)
    loss, recons_loss, zero_abl_loss = losses.mean(0).tolist()

    print(f"loss: {loss:.4f}, recons_loss: {recons_loss:.4f}, zero_abl_loss: {zero_abl_loss:.4f}")
    score = ((zero_abl_loss - recons_loss)/(zero_abl_loss - loss))
    print(f"Reconstruction Score: {score:.2%}")
    # print(f"{((zero_abl_loss - mean_abl_loss)/(zero_abl_loss - loss)).item():.2%}")
    return score, loss, recons_loss, zero_abl_loss