'''
Created on Apr. 12, 2022

@author: yfang
'''
import torch
from torch._C import NoopLogger
import torch.nn



class PrefixCrossAttentionEncoder(torch.nn.Module):
    r'''
    The torch.nn model to encode the prefix

    Input shape: (batch-size, prefix-length)

    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    '''
    def __init__(self, config):
        super().__init__()
        self.prefix_projection = config.prefix_projection
        self.num_hidden_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads
        self.pre_seq_len = config.pre_seq_len
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(config.pre_seq_len, config.hidden_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(config.hidden_size, config.prefix_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(config.prefix_hidden_size, config.num_hidden_layers * 2 * config.hidden_size)
            )
        else:
            self.embedding = torch.nn.Embedding(config.pre_seq_len, config.num_hidden_layers * 2 * config.hidden_size)

    def forward(self, prefix: torch.Tensor, graph_prefix: torch.Tensor):
        """
        prefix: [batch_size * num_choices, seq_len]
        graph_prefix: [batch_size * num_choices, graph_seq_len, num_layers, n_embd]
        """
        assert (graph_prefix.size()[3] == self.n_embd and graph_prefix.size()[0] == prefix.size()[0])
        graph_key_values = torch.cat([graph_prefix for _ in range(2 * self.n_head)],3)
        # graph_key_values: [batch_size * num_choices, graph_seq_len, num_layers, 2 * hidden_size]
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            prefix_key_values = self.trans(prefix_tokens)
        else:
            prefix_key_values = self.embedding(prefix)
        # prefix_key_values: [batch_size * num_choices, seq_len, num_hidden_layers * 2 * hidden_size]
        prefix_key_values = prefix_key_values.view(*prefix_key_values.size()[:2], self.num_hidden_layers, 2 * self.hidden_size)
        # prefix_key_values: [batch_size * num_choices, seq_len, num_hidden_layers, 2 * hidden_size]
        p = prefix_key_values[:,:prefix_key_values.size(1)-graph_key_values.size(1),:,:]
        s = prefix_key_values[:,prefix_key_values.size(1)-graph_key_values.size(1):,:,:]
        s = s[:,:,:prefix_key_values.size(2)-graph_key_values.size(2),:]
        s = torch.cat([s, graph_key_values], dim=2)
        past_key_values = torch.cat([p,s], dim=1)
        past_key_values = past_key_values.view(-1, self.pre_seq_len, self.num_hidden_layers * 2 * self.hidden_size)
        return past_key_values

