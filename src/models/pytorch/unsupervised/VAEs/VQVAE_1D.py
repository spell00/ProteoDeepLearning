from src.utils.stochastic import GaussianSample
from torch.autograd import Variable
from src.models.pytorch.unsupervised.VAEs.VectorQuantizer import VectorQuantizer
from src.utils.distributions import log_gaussian, log_standard_gaussian
from src.utils.flow import NormalizingFlows
from src.utils.masked_layer import GatedConv1d, GatedConvTranspose1d
import torch
from torch import nn

in_channels = None
out_channels = None
kernel_sizes = None
strides = None


class Stochastic(nn.Module):
    """
    Base stochastic layer that uses the
    reparametrization trick [Kingma 2013]
    to draw a sample from a distribution
    parametrised by mu and log_var.
    """

    def reparametrize(self, mu, log_var):
        epsilon = Variable(torch.randn(mu.size()), requires_grad=False)

        if mu.is_cuda:
            epsilon = epsilon.cuda()

        # log_std = 0.5 * log_var
        # std = exp(log_std)
        std = log_var.mul(0.5).exp_()

        # z = std * epsilon + mu
        z = mu.addcmul(std, epsilon)

        return z


class VariationalAutoencoderVQ(torch.nn.Module):
    def __init__(self, z_dim, in_channels, out_channels, kernel_sizes, strides, dilatations, num_embeddings,
                 flow_type="nf",
                 n_flows=2, gated=True):
        super(VariationalAutoencoderVQ, self).__init__()
        self.conv_layers = []
        self.deconv_layers = []
        self.bns = []
        self.bns_decoder = []
        self.GaussianSample = GaussianSample(z_dim, z_dim)
        self.relu = torch.nn.PReLU()
        for ins, outs, ksize, stride, dilats in zip(in_channels, out_channels, kernel_sizes, strides, dilatations):
            if not gated:
                self.conv_layers += [torch.nn.Conv1d(in_channels=ins, out_channels=outs, kernel_size=ksize,
                                                     stride=stride, dilation=dilats)]
            else:
                self.conv_layers += [GatedConv1d(in_channels=ins, out_channels=outs, kernel_size=ksize, stride=stride,
                                                 padding=0, dilation=dilats, activation=nn.Tanh())]

            self.bns += [nn.BatchNorm1d(num_features=outs).cuda()]

        for ins, outs, ksize, stride, dilats in zip(reversed(out_channels), reversed(in_channels),
                                                    reversed(kernel_sizes), reversed(strides), dilatations):
            if not gated:
                self.deconv_layers += [torch.nn.ConvTranspose1d(in_channels=ins, out_channels=outs,
                                                                kernel_size=ksize, stride=stride, dilation=dilats)]
            else:
                self.deconv_layers += [GatedConvTranspose1d(in_channels=ins, out_channels=outs, kernel_size=ksize,
                                                            stride=stride, padding=0, dilation=dilats,
                                                            activation=nn.Tanh())]

            self.bns_decoder += [nn.BatchNorm1d(num_features=outs).cuda()]
        self._vq = VectorQuantizer(
            num_embeddings=64,
            embedding_dim=z_dim,
            commitment_cost=0.25,
            device='cuda'
        )

        self.dense1 = torch.nn.Linear(in_features=391, out_features=z_dim)
        self.dense2 = torch.nn.Linear(in_features=z_dim, out_features=391)
        self.dense1_bn = nn.BatchNorm1d(num_features=z_dim)
        self.dense2_bn = nn.BatchNorm1d(num_features=391)
        self.dropout = nn.Dropout(0.2)
        self.conv_layers = nn.ModuleList(self.conv_layers)
        self.deconv_layers = nn.ModuleList(self.deconv_layers)
        self.flow_type = flow_type
        self.n_flows = n_flows
        if self.flow_type == "nf":
            self.flow = NormalizingFlows(in_features=[z_dim], n_flows=n_flows)

    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _kld(self, z, mu, log_var):
        if self.flow_type == "nf" and self.n_flows > 0:
            f_z, log_det_z = self.flow(z)
            qz = log_gaussian(z, mu, log_var) - sum(log_det_z)
            z = f_z
        else:
            qz = log_gaussian(z, mu, log_var)
        pz = log_standard_gaussian(z)

        kl = qz - pz

        return kl

    def encoder(self, x):
        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x)
            # x = self.relu(x)
            # x = self.bns[i](x)
            # x = self.dropout(x)
        x = x.squeeze()
        z = self.dense1(x)
        z = self.dense1_bn(z)
        z = self.relu(z)
        z = self.dropout(z)

        return z

    def decoder(self, z):
        z = self.dense2(z)
        z = self.dense2_bn(z).unsqueeze(1)
        x = self.relu(z)
        x = self.dropout(x)

        for i in range(len(self.deconv_layers)):
            x = self.deconv_layers[i](x)
            #  = self.relu(x)
            # x = self.bns_decoder[i](x)
            # x = self.dropout(x)
        # TODO there is 139 / 300000 loss at the end, close enough?

        return x

    def forward(self, x):

        z = self.encoder(x)
        # z = self._pre_vq_conv(z)
        # Vector Quantization
        vq_loss, quantized, perplexity, _, _, encoding_indices, \
        losses, _, _, _, concatenated_quantized = self._vq(z.unsqueeze(1),
                                                           record_codebook_stats=False,
                                                           compute_distances_if_possible=False)

        x = self.decoder(quantized)
        probs = torch.sigmoid(x)
        return probs, vq_loss

    def sample(self, z, y=None):
        """
        Given z ~ N(0, I) generates a sample from
        the learned distribution based on p_θ(x|z).
        :param z: (torch.autograd.Variable) Random normal variable
        :return: (torch.autograd.Variable) generated sample
        """
        return self.decoder(z)

    def get_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data.shape)

    def get_total_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data.shape)
