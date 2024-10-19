import yaml
import argparse
import torch
import random
import os
import numpy as np
from tqdm import tqdm
from vqvae import get_model
from torch.utils.data.dataloader import DataLoader
from src.perturbation_module.diffusion_model.dataset.dataset_uncond import SciplexDatasetUncond
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_for_one_epoch(epoch_idx, model, sciplex_loader, optimizer, crtierion, config):
    r"""
    Method to run the training for one epoch.
    :param epoch_idx: iteration number of current epoch
    :param model: VQVAE model
    :param sciplex_loader: Data loder for mnist
    :param optimizer: optimzier to be used taken from config
    :param crtierion: For computing the loss
    :param config: configuration for the current run
    :return:
    """
    recon_losses = []
    codebook_losses = []
    commitment_losses = []
    losses = []
    # We ignore the label for VQVAE
    count = 0
    for im in tqdm(sciplex_loader):
        im = im.float().to(device)
        optimizer.zero_grad()
        model_output = model(im)
        output = model_output['generated_output']
        quantize_losses = model_output['quantized_losses']

        recon_loss = crtierion(output, im)
        loss = (config['train_params']['reconstruction_loss_weight']*recon_loss +
                config['train_params']['codebook_loss_weight']*quantize_losses['codebook_loss'] +
                config['train_params']['commitment_loss_weight']*quantize_losses['commitment_loss'])
        recon_losses.append(recon_loss.item())
        codebook_losses.append(config['train_params']['codebook_loss_weight']*quantize_losses['codebook_loss'].item())
        commitment_losses.append(quantize_losses['commitment_loss'].item())
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    print('Finished epoch: {} | Recon Loss : {:.4f} | Codebook Loss : {:.4f} | Commitment Loss : {:.4f}'.
          format(epoch_idx + 1,
                 np.mean(recon_losses),
                 np.mean(codebook_losses),
                 np.mean(commitment_losses)))
    return np.mean(losses)


def train(args):
    ######## Read the config file #######
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    #######################################

    ######## Set the desired seed value #######
    seed = config['train_params']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    #######################################

    # Create the model and dataset
    model = get_model(config).to(device)
    sciplex = SciplexDatasetUncond(config['train_params']['train_adata_path'])
    sciplex_loader = DataLoader(sciplex, batch_size=config['train_params']['batch_size'], shuffle=True, num_workers=0)
    num_epochs = config['train_params']['epochs']
    optimizer = Adam(model.parameters(), lr=config['train_params']['lr'])
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=1, verbose=True)
    criterion = {
        'l1': torch.nn.L1Loss(),
        'l2': torch.nn.MSELoss()
    }.get(config['train_params']['crit'])

    # Create output directories
    # if not os.path.exists(config['train_params']['task_name']):
    #     os.mkdir(config['train_params']['task_name'])
    # if not os.path.exists(os.path.join(config['train_params']['task_name'],
    #                                    config['train_params']['output_train_dir'])):
    #     os.mkdir(os.path.join(config['train_params']['task_name'],
    #                           config['train_params']['output_train_dir']))

    # # Load checkpoint if found
    # if os.path.exists(os.path.join(config['train_params']['task_name'],
    #                                                     config['train_params']['ckpt_name'])):
    #     print('Loading checkpoint')
    #     model.load_state_dict(torch.load(os.path.join(config['train_params']['task_name'],
    #                                                   config['train_params']['ckpt_name']), map_location=device))
    best_loss = np.inf

    for epoch_idx in range(num_epochs):
        mean_loss = train_for_one_epoch(epoch_idx, model, sciplex_loader, optimizer, criterion, config)
        scheduler.step(mean_loss)
        # Simply update checkpoint if found better version
        if mean_loss < best_loss:
            print('Improved Loss to {:.4f} .... Saving Model'.format(mean_loss))
            torch.save(model.state_dict(), os.path.join(config['train_params']['task_name'],
                                                        config['train_params']['ckpt_name']))
            best_loss = mean_loss
        else:
            print('No Loss Improvement')


if __name__ == '__main__':
    ROOT = 'C:\\Users\\curea\\Documents\\bioFM for drug discovery\\dege-fm\\'
    parser = argparse.ArgumentParser(description='Arguments for vq vae training')
    parser.add_argument('--config', dest='config_path',
                        default=ROOT+'config\\sciplex_uce_emb.yaml', type=str)
    args = parser.parse_args()
    train(args)