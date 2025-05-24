import torch
import numpy as np
import torch.nn as nn



class PRnet(nn.Module):
    """Model for PRnet class.
    """

    def __init__(self, gene_vector_dim=5000, hidden_layer_sizes: list = [128, 64], agg_latent_dim: int = 10,
                 adaptor_layer_sizes: list = [128], drug_latent_dim: int = 50, comb_num: int = 1,
                 drug_dimension: int = 1024, dr_rate: float = 0.05, noise_dimension: int = 10, add_relu=True):
        super().__init__()

        self.x_dim_ = gene_vector_dim  # adata.n_vars
        self.c_dim = drug_latent_dim
        self.n_dim = noise_dimension
        self.z_dim_ = agg_latent_dim
        self.hidden_layer_sizes_ = hidden_layer_sizes
        self.dr_rate_ = dr_rate
        self.adaptor_layer_sizes = adaptor_layer_sizes
        self.comb_adapt_dim = comb_num * drug_dimension

        self.PGM = PGM(self.x_dim_, self.c_dim, self.n_dim, self.hidden_layer_sizes_, self.z_dim_,
                       self.adaptor_layer_sizes, self.comb_adapt_dim, self.dr_rate_, add_relu=add_relu)

        self.is_trained_ = False
        self.trainer = None

    def get_latent(self, x: torch.Tensor, c: torch.Tensor, n: torch.Tensor):
        """
        Map chemical perturbation and unperturbed state in to the latent space. This function will feed data
        in Perturb-encoder and compute the latent space coordinates
        for each sample in data.
        """
        latent = self.PGM.get_latent(x, c, n)
        latent = latent.cpu().detach()  # to cpu then detach from the comput.graph
        return np.array(latent)

    def get_per_latent(self, x: torch.Tensor, c: torch.Tensor, n: torch.Tensor):
        """
        Map chemical perturbation in to the latent space.
        """

        latent = self.PGM.get_per_latent(x, c, n)
        latent = latent.cpu().detach()  # to cpu then detach from the comput.graph
        return np.array(latent)

    def get_PGM(self):
        return self.PGM


class PGM(nn.Module):
    """perturbation-conditioned generative model involves  three  key  components:  the  Perturb-adaptor,  the  Perturb-encoder,  and  the Perturb-decoder.

    """

    def __init__(self, gene_vector_dim: int, drug_latent_dim: int, noise_dim: int, hidden_layer_sizes: list = [128, 128], agg_latent_dim: int = 10,
                 adaptor_layer_sizes: list = [128], drug_initial_dim: int = 1024, dr_rate: float = 0.05, add_relu=True, **kwargs):
        super().__init__()
        assert isinstance(hidden_layer_sizes, list)
        assert isinstance(agg_latent_dim, int)

        self.x_dim = gene_vector_dim   #Gene expression vector length
        self.c_dim = drug_latent_dim   #Chemical structure latent embedding
        self.n_dim = noise_dim   #noise dimension
        self.z_dim = agg_latent_dim #Latent dimension of x + c + n
        self.hidden_layer_sizes = hidden_layer_sizes #Hidden layers of decoder and encoder
        self.dr_rate = dr_rate   #dropout
        self.adaptor_layer_sizes = adaptor_layer_sizes #hidden layers of adaptor
        self.comb_adapt_dim = drug_initial_dim  #chemical fingerprint size

        encoder_layer_sizes = self.hidden_layer_sizes.copy()
        encoder_layer_sizes.insert(0, self.x_dim + self.c_dim)

        decoder_layer_sizes = self.hidden_layer_sizes.copy()
        decoder_layer_sizes.reverse()
        decoder_layer_sizes.append(self.x_dim * 2)  #predicitng mu + std of original vector

        adaptor_layer_sizes_ = self.adaptor_layer_sizes.copy()
        adaptor_layer_sizes_.insert(0, self.comb_adapt_dim)

        self.encoder = PEncoder(encoder_layer_sizes, self.z_dim, self.dr_rate)
        self.decoder = PDecoder(self.z_dim + self.c_dim + self.n_dim, decoder_layer_sizes, self.x_dim, self.dr_rate, add_relu=add_relu)
        self.CombAdaptor = PAdaptor(adaptor_layer_sizes_, self.c_dim, self.dr_rate)

    def get_latent(self, x: torch.Tensor, c: torch.Tensor, n: torch.Tensor):
        """
        Map chemical perturbation and unperturbed state in to the latent space. This function will feed data
        in Perturb-encoder and compute the latent space coordinates
        for each sample in data.
        """

        c = self.CombAdaptor(c)
        noise = (x, c)
        input_ = torch.cat(noise, 1)
        latent = self.encoder(input_)

        return latent

    def get_per_latent(self, x: torch.Tensor, c: torch.Tensor, n: torch.Tensor):
        """
        Map `data` in to the latent space. This function will feed data
        in encoder part of VAE and compute the latent space coordinates
        for each sample in data.
        """
        latent = self.CombAdaptor(c)
        return latent

    def forward(self, x: torch.Tensor, c: torch.Tensor, n: torch.Tensor):
        c = self.CombAdaptor(c)
        noise = (x, c)
        input_ = torch.cat(noise, 1)
        z = self.encoder(input_)
        z_c = (z, c, n)
        z_c_ = torch.cat(z_c, 1)
        x_hat = self.decoder(z_c_)
        return x_hat


class PEncoder(nn.Module):
    """
    Constructs the  Perturb-encoder. This class implements the encoder part of PRnet. It will transform primary data in the `n_vars` dimension-space and chemical perturbation to `z_dimension` latent space.

    """

    def __init__(self, layer_sizes: list, z_dimension: int, dropout_rate: float):
        super().__init__()  # to run nn.Module's init method

        # encoder architecture
        self.FC = None
        if len(layer_sizes) > 1:
            self.FC = nn.Sequential()
            for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
                if i == 0:
                    self.FC.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size, bias=False))
                else:
                    self.FC.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
                    self.FC.add_module("N{:d}".format(i), module=nn.BatchNorm1d(out_size))
                    self.FC.add_module(name="A{:d}".format(i), module=nn.LeakyReLU(negative_slope=0.3))
                    self.FC.add_module(name="D{:d}".format(i), module=nn.Dropout(p=dropout_rate))

        # self.FC = nn.ModuleList(self.FC)

        self.mean_encoder = nn.Linear(layer_sizes[-1], z_dimension)

    def forward(self, x: torch.Tensor):
        if self.FC is not None:
            x = self.FC(x)
        mean = self.mean_encoder(x)

        return mean


class PDecoder(nn.Module):
    """
            Constructs the  Perturb-decoder.  Decodes data from latent space to data space. It will transform constructed latent space to the previous space of data with means and log variances of n_dimensions = n_vars.
        """

    def __init__(self, z_dimension: int, layer_sizes: list, x_dimension: int, dropout_rate: float, add_relu=True):
        super().__init__()

        layer_sizes = [z_dimension] + layer_sizes
        # decoder architecture
        # Create first Decoder layer
        self.FirstL = nn.Sequential()
        self.FirstL.add_module(name="L0", module=nn.Linear(layer_sizes[0], layer_sizes[1], bias=False))
        self.FirstL.add_module("N0", module=nn.BatchNorm1d(layer_sizes[1]))
        self.FirstL.add_module(name="A0", module=nn.LeakyReLU(negative_slope=0.3))
        self.FirstL.add_module(name="D0", module=nn.Dropout(p=dropout_rate))

        self.add_relu = add_relu

        # Create all Decoder hidden layers
        if len(layer_sizes) > 2:
            self.HiddenL = nn.Sequential()
            for i, (in_size, out_size) in enumerate(zip(layer_sizes[1:-1], layer_sizes[2:])):
                if i + 3 < len(layer_sizes):
                    self.HiddenL.add_module(name="L{:d}".format(i + 1), module=nn.Linear(in_size, out_size, bias=False))
                    self.HiddenL.add_module("N{:d}".format(i + 1), module=nn.BatchNorm1d(out_size, affine=True))
                    self.HiddenL.add_module(name="A{:d}".format(i + 1), module=nn.LeakyReLU(negative_slope=0.3))
                    self.HiddenL.add_module(name="D{:d}".format(i + 1), module=nn.Dropout(p=dropout_rate))
        else:
            self.HiddenL = None

        # Create Output Layers
        self.recon_decoder = nn.Sequential(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
        if self.add_relu:
            self.relu = nn.ReLU()

    def forward(self, z: torch.Tensor):
        dec_latent = self.FirstL(z)

        # Compute Hidden Output
        if self.HiddenL is not None:
            x = self.HiddenL(dec_latent)
        else:
            x = dec_latent

        # Compute Decoder Output
        recon_x = self.recon_decoder(x)
        dim = recon_x.size(1) // 2

        if self.add_relu:
            recon_x = torch.cat((self.relu(recon_x[:, :dim]), recon_x[:, dim:]), dim=1)
        else:
            recon_x = torch.cat((recon_x[:, :dim], recon_x[:, dim:]), dim=1)
        return recon_x


class PAdaptor(nn.Module):
    """
    Constructs the  Perturb-adaptor. This class implements the adaptor part of PRnet. It will chemical perturbation in to 'comb_num' latent space.

    """

    def __init__(self, layer_sizes: list, comb_dimension: int, dropout_rate: float):
        super().__init__()  # to run nn.Module's init method

        # encoder architecture
        self.FC = None
        if len(layer_sizes) > 1:
            self.FC = nn.Sequential()
            for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
                if i == 0:
                    self.FC.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size, bias=False))
                else:
                    self.FC.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
                    self.FC.add_module("N{:d}".format(i), module=nn.BatchNorm1d(out_size))
                    self.FC.add_module(name="A{:d}".format(i), module=nn.LeakyReLU(negative_slope=0.3))
                    self.FC.add_module(name="D{:d}".format(i), module=nn.Dropout(p=dropout_rate))

        self.comb_encoder = nn.Linear(layer_sizes[-1], comb_dimension)

    def forward(self, x: torch.Tensor):
        if self.FC is not None:
            x = self.FC(x)
        comb_encode = self.comb_encoder(x)
        return comb_encode