import torch
import numpy as np

class LineInvariantAEMultipleDecoders(torch.nn.Module):
    """Class for auto-encoders with two decoders. This class does not contain a forward function."""
    def __init__(self,
                 column_encoder_dims,
                 column_decoder_dims,
                 line_encoder_dims,
                 line_decoder_dims,
                 number_decoders,
                 n_lines,
                 n_columns_seeds=4,
                 dropout=0):
        """Initialise auto encoder with hyperbolic tangent activation function

        :param encoder_dims:    list, List of dimensions for encoder, including input/output layers
        :param decoder_dims:    list, List of dimensions for decoder, including input/output layers
        :param dropout:         int, value of the dropout probability
        """
        self.n_lines = n_lines
        self.n_columns_seeds = n_columns_seeds
        self.n_columns = column_encoder_dims[0]
        self.bottleneck_dim = line_encoder_dims[-1]
        super(LineInvariantAEMultipleDecoders, self).__init__()
        layers = []
        for i in range(len(line_encoder_dims) - 2):
            layers.append(torch.nn.Linear(line_encoder_dims[i], line_encoder_dims[i + 1]))
            layers.append(torch.nn.Dropout(dropout))
            layers.append(torch.nn.Tanh())
        layers.append(torch.nn.Linear(line_encoder_dims[-2], line_encoder_dims[-1]))
        self.line_encoder = torch.nn.Sequential(*layers)

        layers = []
        for i in range(len(column_encoder_dims) - 2):
            layers.append(torch.nn.Linear(column_encoder_dims[i], column_encoder_dims[i + 1]))
            layers.append(torch.nn.Dropout(dropout))
            layers.append(torch.nn.Tanh())
        layers.append(torch.nn.Linear(column_encoder_dims[-2], column_encoder_dims[-1]))
        self.column_encoder = torch.nn.Sequential(*layers)

        self.line_decoders = []
        for i in range(number_decoders):
            layers = []
            for i in range(len(line_decoder_dims) - 2):
                layers.append(torch.nn.Linear(line_decoder_dims[i], line_decoder_dims[i + 1]))
                layers.append(torch.nn.Dropout(dropout))
                layers.append(torch.nn.Tanh())
            layers.append(torch.nn.Linear(line_decoder_dims[-2], line_decoder_dims[-1]))
            self.line_decoders.append(torch.nn.Sequential(*layers))
        self.line_decoders = torch.nn.ModuleList(self.line_decoders)

        self.column_decoders = []
        for i in range(number_decoders):
            layers = []
            for i in range(len(column_decoder_dims) - 2):
                layers.append(torch.nn.Linear(column_decoder_dims[i], column_decoder_dims[i + 1]))
                layers.append(torch.nn.Dropout(dropout))
                layers.append(torch.nn.Tanh())
            layers.append(torch.nn.Linear(column_decoder_dims[-2], column_decoder_dims[-1]))
            self.column_decoders.append(torch.nn.Sequential(*layers))
        self.column_decoders = torch.nn.ModuleList(self.column_decoders)

    def decoded(self, inp, dec_index):
        enc = self.encoded(inp).reshape([len(inp), 1, self.bottleneck_dim])
        dec = torch.transpose(self.line_decoders[dec_index](enc), dim0=-2, dim1=-1)
        dec = dec.reshape([len(inp), self.n_lines, self.n_columns_seeds])
        dec = self.column_decoders[dec_index](dec)
        return dec

    def encoded(self, inp):
        return self.line_encoder(torch.mean(self.column_encoder(inp), dim=-2))


