import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import utils
from models.vqvae import VQVAE
from tqdm import tqdm, trange

parser = argparse.ArgumentParser()

"""
Hyperparameters
"""
timestamp = utils.readable_timestamp()

parser.add_argument("--batch_size", type=int, default=32) # Check
parser.add_argument("--n_updates", type=int, default=50000) # Check
parser.add_argument("--n_hiddens", type=int, default=128) # Check
parser.add_argument("--n_residual_hiddens", type=int, default=32) # Check
parser.add_argument("--n_residual_layers", type=int, default=2) # Check
parser.add_argument("--embedding_dim", type=int, default=64) # Check
parser.add_argument("--n_embeddings", type=int, default=512) # Check
parser.add_argument("--beta", type=float, default=0.25) # Check
parser.add_argument("--learning_rate", type=float, default=3e-4) # Check
parser.add_argument("--log_interval", type=int, default=50)
parser.add_argument("--dataset",  type=str, default='CIFAR10') # Check

# whether or not to save model
parser.add_argument("-save", action="store_true")
parser.add_argument("--filename",  type=str, default=timestamp)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.save:
    print('Results will be saved in ./results/vqvae_' + args.filename + '.pt')

""" 
Load data and define batch data loaders
"""

training_data, validation_data, training_loader, validation_loader, x_train_var = utils.load_data_and_data_loaders(
    args.dataset, args.batch_size)
"""
Set up VQ-VAE model with components defined in ./models/ folder
"""

model = VQVAE(args.n_hiddens, args.n_residual_hiddens,
              args.n_residual_layers, args.n_embeddings, args.embedding_dim, args.beta).to(device)

"""
Set up optimizer and training loop
"""
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)

model.train()

results = {
    'n_updates': 0,
    'recon_errors': [],
    'loss_vals': [],
    'perplexities': [],
}


def train():


    while results['n_updates'] < args.n_updates:
        
        for x, _ in training_loader:

            # This only gives the first batch of data each time?
            x = x.to(device)
            optimizer.zero_grad()

            # Hmm x is coming from the training_loader and is never normalised? 
            # Original converts images to floating point with range [-0.5, 0.5]
            embedding_loss, x_hat, perplexity = model(x)
            recon_loss = torch.mean((x_hat - x)**2) / x_train_var
            loss = recon_loss + embedding_loss

            loss.backward()
            optimizer.step()

            results["recon_errors"].append(recon_loss.cpu().detach().numpy())
            results["perplexities"].append(perplexity.cpu().detach().numpy())
            results["loss_vals"].append(loss.cpu().detach().numpy())
            results["n_updates"] += 1

            if results["n_updates"] % args.log_interval == 0:
                """
                save model and print values
                """
                if args.save:
                    hyperparameters = args.__dict__
                    utils.save_model_and_results(
                        model, results, hyperparameters, args.filename)

                print('Update #', results["n_updates"], 'Recon Error:',
                    np.mean(results["recon_errors"][-args.log_interval:]),
                    'Loss', np.mean(results["loss_vals"][-args.log_interval:]),
                    'Perplexity:', np.mean(results["perplexities"][-args.log_interval:]))

                # TODO update to log to tensorboard, also 

if __name__ == "__main__":
    train()
