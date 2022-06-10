import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import get_hyperparams, get_config
from modules.glow import Glow, GlowOnGraph

config = get_config()

def gaussian_nll(x, mean, ln_var, reduce='sum'):
    """Computes the negative log-likelihood of a Gaussian distribution.

    Given two variable ``mean`` representing :math:`\\mu` and ``ln_var``
    representing :math:`\\log(\\sigma^2)`, this function computes in
    elementwise manner the negative log-likelihood of :math:`x` on a
    Gaussian distribution :math:`N(\\mu, S)`,

    .. math::

        -\\log N(x; \\mu, \\sigma^2) =
        \\log\\left(\\sqrt{(2\\pi)^D |S|}\\right) +
        \\frac{1}{2}(x - \\mu)^\\top S^{-1}(x - \\mu),

    where :math:`D` is a dimension of :math:`x` and :math:`S` is a diagonal
    matrix where :math:`S_{ii} = \\sigma_i^2`.

    The output is a variable whose value depends on the value of
    the option ``reduce``. If it is ``'no'``, it holds the elementwise
    loss values. If it is ``'sum'``, loss values are summed up.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.
        mean (:class:`~chainer.Variable` or :ref:`ndarray`): A variable
            representing mean of a Gaussian distribution, :math:`\\mu`.
        ln_var (:class:`~chainer.Variable` or :ref:`ndarray`): A variable
            representing logarithm of variance of a Gaussian distribution,
            :math:`\\log(\\sigma^2)`.
        reduce (str): Reduction option. Its value must be either
            ``'sum'`` or ``'no'``. Otherwise, :class:`ValueError` is raised.

    Returns:
        ~chainer.Variable:
            A variable representing the negative log-likelihood.
            If ``reduce`` is ``'no'``, the output variable holds array
            whose shape is same as one of (hence both of) input variables.
            If it is ``'sum'``, the output variable holds a scalar value.

    """
    if reduce not in ('sum', 'no'):
        raise ValueError(
            "only 'sum' and 'no' are valid for 'reduce', but '%s' is "
            'given' % reduce)

    x_prec = torch.exp(-ln_var)  # 324
    x_diff = x - mean  # (256,324) - (324,) --> (256,324)
    x_power = (x_diff * x_diff) * x_prec * -0.5
    loss = (ln_var + math.log(2 * (math.pi))) / 2 - x_power
    if reduce == 'sum':
        return loss.sum()
    else:
        return loss

def rescale_adj(adj, type='all'):
    if type == 'view':
        out_degree = adj.sum(dim=-1)
        out_degree_sqrt_inv = out_degree.pow(-1)
        out_degree_sqrt_inv[out_degree_sqrt_inv == float('inf')] = 0
        adj_prime = out_degree_sqrt_inv.unsqueeze(-1) * adj  # (256,4,9,1) * (256, 4, 9, 9) = (256, 4, 9, 9)
    else:  # default type all
        num_neighbors = adj.sum(dim=(1, 2)).float()
        num_neighbors_inv = num_neighbors.pow(-1)
        num_neighbors_inv[num_neighbors_inv == float('inf')] = 0
        adj_prime = num_neighbors_inv[:, None, None, :] * adj
    return adj_prime


def logit_pre_process(x, a=0.05, bounds=0.9):
    """Dequantize the input image `x` and convert to logits.

    See Also:
        - Dequantization: https://arxiv.org/abs/1511.01844, Section 3.1
        - Modeling logits: https://arxiv.org/abs/1605.08803, Section 4.1

    Args:
        x (torch.Tensor): Input image.

    Returns:
        y (torch.Tensor): Dequantized logits of `x`.
    """
    y = (1-a) * x + a * torch.rand_like(x)
    y = (2 * y - 1) * bounds
    y = (y + 1) / 2
    y = y.log() - (1. - y).log()

    # Save log-determinant of Jacobian of initial transform
    ldj = F.softplus(y) + F.softplus(-y) \
        - F.softplus(torch.tensor(math.log(1. - bounds) - math.log(bounds)))
    sldj = ldj.flatten(1).sum(-1)

    return y, sldj


class TIDFlow(nn.Module):
    def __init__(self, hyper_params, data_name):
        super(TIDFlow, self).__init__()
        self.hyper_params = hyper_params

        # More parameters derived from hyper_params for easy use
        if data_name == 'qm9':
            self.b_n_type = 4
            self.a_n_node = 9
            self.a_n_type = 5
        self.b_size = self.a_n_node * self.a_n_node * self.b_n_type
        self.a_size = self.a_n_node * self.a_n_type
        self.noise_scale = hyper_params.noise_scale
        if hyper_params.learn_dist:
            self.ln_var = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer('ln_var', torch.zeros(1))

        self.bond_model = Glow(
            in_channel=self.b_n_type,
            n_flow=hyper_params.b_n_flow,
            n_block=hyper_params.b_n_block,
            squeeze_fold=hyper_params.b_n_squeeze,
            hidden_channel=hyper_params.b_hidden_ch,
            affine=hyper_params.b_affine,
            conv_lu=hyper_params.b_conv_lu
        )

        self.atom_model = GlowOnGraph(
            n_node=self.a_n_node,
            in_dim=self.a_n_type,
            hidden_dim_dict={'gnn': hyper_params.a_hidden_gnn, 'linear': hyper_params.a_hidden_lin},
            n_flow=hyper_params.a_n_flow,
            n_block=hyper_params.a_n_block,
            mask_row_size_list=hyper_params.mask_row_size_list,
            mask_row_stride_list=hyper_params.mask_row_stride_list,
            affine=hyper_params.a_affine
        )

    def forward(self, adj, x, adj_normalized):
        """
        :param adj:  (256,4,9,9)
        :param x: (256,9,5)
        :return:
        """
        h = x
        if self.training:
            if self.noise_scale == 0:
                h = h/2.0 - 0.5 + torch.rand_like(x) * 0.4
            else:
                h = h + torch.rand_like(x) * self.noise_scale
        h, sum_log_det_jacs_x = self.atom_model(adj_normalized, h)

        # add uniform noise to adjacency tensors
        if self.training:
            if self.noise_scale == 0:
                adj = adj/2.0 - 0.5 + torch.rand_like(adj) * 0.4
            else:
                adj = adj + torch.rand_like(adj) * self.noise_scale
        adj_h, sum_log_det_jacs_adj = self.bond_model(adj)
        out = [h, adj_h]

        return out, [sum_log_det_jacs_x, sum_log_det_jacs_adj]

    def reverse(self, z, true_adj=None):
        """
        Returns a molecule, given its latent vector.
        :param z: latent vector. Shape: [B, N*N*M + N*T]    (100,369) 369=9*9 * 4 + 9*5
            B = Batch size, N = number of atoms, M = number of bond types,
            T = number of atom types (Carbon, Oxygen etc.)
        :param true_adj: used for testing. An adjacency matrix of a real molecule
        :return: adjacency matrix and feature matrix of a molecule
        """
        batch_size = z.shape[0]

        with torch.no_grad():
            z_x = z[:, :self.a_size]
            z_adj = z[:, self.a_size:]

            if true_adj is None:
                h_adj = z_adj.reshape(batch_size, self.b_n_type, self.a_n_node, self.a_n_node)
                h_adj = self.bond_model.reverse(h_adj)

                if self.noise_scale == 0:
                    h_adj = (h_adj + 0.5) * 2
                # decode adjacency matrix from h_adj
                adj = h_adj
                adj = adj + adj.permute(0, 1, 3, 2)
                adj = adj / 2
                adj = adj.softmax(dim=1)
                max_bond = adj.max(dim=1).values.reshape(batch_size, -1, self.a_n_node, self.a_n_node)
                adj = torch.floor(adj / max_bond)
            else:
                adj = true_adj

            h_x = z_x.reshape(batch_size, self.a_n_node, self.a_n_type)
            adj_normalized = rescale_adj(adj).to(h_x)
            h_x = self.atom_model.reverse(adj_normalized, h_x)
            if self.noise_scale == 0:
                h_x = (h_x + 0.5) * 2
        return adj, h_x

    def log_prob(self, z, logdet):
        z[0] = z[0].reshape(z[0].shape[0],-1)
        z[1] = z[1].reshape(z[1].shape[0], -1)

        logdet[0] = logdet[0] - self.a_size * math.log(2.)
        logdet[1] = logdet[1] - self.b_size * math.log(2.)
        if len(self.ln_var) == 1:
            ln_var_adj = self.ln_var * torch.ones([self.b_size]).to(z[0])
            ln_var_x = self.ln_var * torch.ones([self.a_size]).to(z[0])
        else:
            ln_var_adj = self.ln_var[0] * torch.ones([self.b_size]).to(z[0])
            ln_var_x = self.ln_var[1] * torch.ones([self.a_size]).to(z[0])
        nll_adj = torch.mean(
            torch.sum(gaussian_nll(z[1], torch.zeros(self.b_size).to(z[0]), ln_var_adj, reduce='no'), dim=1)
            - logdet[1])
        nll_adj = nll_adj / (self.b_size * math.log(2.))  # the negative log likelihood per dim with log base 2

        nll_x = torch.mean(torch.sum(
            gaussian_nll(z[0], torch.zeros(self.a_size).to(z[0]), ln_var_x, reduce='no'),
            dim=1) - logdet[0])
        nll_x = nll_x / (self.a_size * math.log(2.))
        if nll_x.item() < 0:
            print('nll_x:{}'.format(nll_x.item()))

        return [nll_x, nll_adj]

    def save_hyperparams(self, path):
        self.hyper_params.save(path)


if __name__ == '__main__':
    hyperparams = get_hyperparams()
    with torch.autograd.set_detect_anomaly(True):
        model = TIDFlow(hyperparams)
        bs = 2
        x = torch.ones((bs, 9, 5), dtype=torch.float32)
        adj = torch.ones((bs, 4, 9, 9), dtype=torch.float32)
        output = model(adj, x, adj)
        print(x, adj)
        print("Test forward:", output[0][0].shape,  output[0][1].shape)

        nll_x, nll_adj = model.log_prob(output[0], output[1])
        o = nll_x + nll_adj
        print("Test log_prob:", nll_x, nll_adj, o)
        o.backward()

        z = torch.randn(2, 369)
        r_out = model.reverse(z)
        print("Test reverse:", r_out[0].shape, r_out[1].shape)
        print(r_out[0], r_out[1])




