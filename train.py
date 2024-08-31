import torch
import numpy as np
import pprint
import json
from transformer_lens import utils
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import HfApi
from IPython.display import HTML
import pandas as pd
from functools import partial
from model import AutoEncoder
# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import os
import random
import numpy as np
import torch.optim as optim
import time
from torch.utils.tensorboard import SummaryWriter

model_name = 'models/language_model.pt'
writer = SummaryWriter('runs/language_model')

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
language_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-fr")

# LLM MODEL
d_hidden = 
batch_size = 24

# ENCONDER MODEL
encoder = AutoEncoder(
    d_hidden=d_hidden, 
    version='new',
    d_mlp=2048, 
    dtype=torch.float16, 
    l1_coeff=3e-4
)


# DATA
data = load_dataset("NeelNanda/c4-code-20k", split="train")
tokenized_data = utils.tokenize_and_concatenate(data, tokenizer, max_length=128)
tokenized_data = tokenized_data.shuffle(42)
all_tokens = tokenized_data["tokens"]


# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(encoder.parameters(), lr=1e-6, weight_decay=1e-2)


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def train_model(encoder, language_model, num_epochs, optimizer, data): 
    device = get_device()
    since = time.time()


    best_val_loss = 100_000
    best_epoch = 0
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')

        encoder.train()  # Set model to training mode

        running_loss = 0.0

        # Iterate over data.
        for inputs, _ in data:
            inputs = inputs.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            _, cache = language_model.run_with_cache(inputs, stop_at_layer=1, names_filter=utils.get_act_name("post", 0))
            mlp_acts = cache[utils.get_act_name("post", 0)]
            x = mlp_acts.reshape(-1, d_mlp)
            x_reconstruct = encoder(x)

            loss = encoder.calculate_loss(x, x_reconstruct)

            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
        

        epoch_loss = running_loss / len(data)

        if epoch_loss <= best_val_loss:
            best_val_loss = epoch_loss
            best_epoch = epoch


        writer.add_scalars(
            'loss',
            epoch_loss,
            epoch
        )


    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f"Best validation loss: {best_val_loss} at epoch: {best_epoch}")

    torch.save(encoder.state_dict(), model_name)
    # load best model weights
    # model.load_state_dict(torch.load(best_model_params_path))
    writer.flush()



# RUN
# tokens = all_tokens[:batch_size]
# _, cache = model.run_with_cache(tokens, stop_at_layer=1, names_filter=utils.get_act_name("post", 0))
# mlp_acts = cache[utils.get_act_name("post", 0)]
# mlp_acts_flattened = mlp_acts.reshape(-1, d_mlp)
# loss, x_reconstruct, hidden_acts, l2_loss, l1_loss = encoder(mlp_acts_flattened)
# # This is equivalent to:
# # hidden_acts = F.relu((mlp_acts_flattened - encoder.b_dec) @ encoder.W_enc + encoder.b_enc)
# print("hidden_acts.shape", hidden_acts.shape)

# token_df = make_token_df(tokens)
# token_df["feature"] = utils.to_numpy(hidden_acts[:, feature_id])
# token_df.sort_values("feature", ascending=False).head(20).style.background_gradient("coolwarm")