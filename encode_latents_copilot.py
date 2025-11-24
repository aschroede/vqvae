import argparse
import torch
import numpy as np
from models.vqvae import VQVAE
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def build_modle_from_ckpt(ckpt, device):
    hparams = ckpt.get('hyperparameters', {})

    model = VQVAE(hparams['n_hiddens'],
                  hparams['n_residual_hiddens'],
                  hparams['n_residual_layers'],
                  hparams['n_embeddings'],
                  hparams['embedding_dim'],
                  hparams['beta']).to(device)

    model.load_state_dict(ckpt['model'])

    # not training it
    model.eval()
    return model

def encode_dataset(model: VQVAE, dataloader, device):
    latent_maps = []

    with torch.no_grad():
        for (x, _) in dataloader:
            x = x.to(device)

            z_e = model.encoder(x)
            z_e = model.pre_quantization_conv(z_e)
            _,_,_,_,min_encoding_indices = model.vector_quantization(z_e)

            # min_encoding_indices shape is (B*H*W, 1)
            B = x.size(0)
            H = z_e.size(2)
            W = z_e.size(3)
            # reshape into (B, H, W)
            e_indices = min_encoding_indices.view(B,H,W).cpu().numpy().astype(np.int32)
            # add channel dim to match (B,1,H,W)
            e_indices = e_indices[:,None,:,:]
            latent_maps.append(e_indices)
    latent_maps = np.concatenate(latent_maps, axis=0)

    return latent_maps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--save_path', default='data/latent_e_indices.npy')
    parser.add_argument('--dataset', default='CIFAR10')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_modle_from_ckpt(ckpt, device)

    # use same transforms as training
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    dataset = datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    latents = encode_dataset(model, loader, device)
    print("Saving latents of shape", latents.shape, 'to', args.save_path)
    np.save(args.save_path, latents)


if __name__ == '__main__':
    main()