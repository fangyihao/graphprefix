'''
Created on Apr. 1, 2022

@author: yfang
'''
from model.gat import GATConvE, make_one_hot

import torch
import torch.nn as nn
from utils import layers
from tqdm import tqdm, trange
from utils import data_utils
import random
import numpy as np
from utils import utils
import argparse
from utils import parser_utils

DECODER_DEFAULT_LR = {
    'csqa': 1e-3,
    'obqa': 3e-4,
    'medqa_usmle': 1e-3,
}

def get_devices(use_cuda):
    """Get the devices to put the data and the model based on whether to use GPUs and, if so, how many of them are available."""
    if torch.cuda.device_count() >= 2 and use_cuda:
        device0 = torch.device("cuda:0")
        device1 = torch.device("cuda:1")
        print("device0: {}, device1: {}".format(device0, device1))
    elif torch.cuda.device_count() == 1 and use_cuda:
        device0 = torch.device("cuda:0")
        device1 = torch.device("cuda:0")
    else:
        device0 = torch.device("cpu")
        device1 = torch.device("cpu")
    return device0, device1


def load_data(args, devices, kg):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.cuda:
        torch.cuda.manual_seed(args.seed)


    #########################################################
    # Construct the dataset
    #########################################################
    dataset = data_utils.DataLoader(args.train_statements, args.train_adj,
        args.dev_statements, args.dev_adj,
        args.test_statements, args.test_adj,
        batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
        device=devices,
        model_name=args.encoder,
        max_node_num=args.max_node_num, max_seq_length=args.max_seq_len,
        is_inhouse=args.inhouse, inhouse_train_qids_path=args.inhouse_train_qids,
        subsample=args.subsample, n_train=args.n_train, debug=args.debug, cxt_node_connects_all=args.cxt_node_connects_all, kg=kg)

    return dataset

def batch_graph(edge_index_init, edge_type_init, n_nodes):
    """
    edge_index_init: list of (n_examples, ). each entry is torch.tensor(2, E)
    edge_type_init:  list of (n_examples, ). each entry is torch.tensor(E, )
    """
    n_examples = len(edge_index_init)
    edge_index = [edge_index_init[_i_] + _i_ * n_nodes for _i_ in range(n_examples)]
    edge_index = torch.cat(edge_index, dim=1) #[2, total_E]
    edge_type = torch.cat(edge_type_init, dim=0) #[total_E, ]
    return edge_index, edge_type

def get_inputs(args):
    
    print("args: {}".format(args))
    
    kg = "cpnet"
    if args.dataset == "medqa_usmle":
        kg = "ddb"
    
    devices = get_devices(args.cuda)
    
    dataset = load_data(args, devices, kg)
    train_dataloader = dataset.train()

    #resize_token_embeddings(len(dataset.tokenizer))
    
    for qids, labels, *input_data in tqdm(train_dataloader, desc="Batch"):
        bs = labels.size(0)
        print("bs:", bs) # 128
        for a in range(0, bs, args.mini_batch_size):
            b = min(a + args.mini_batch_size, bs)
            print("a:", a) # 0
            print("b:", b) # 8
            print("args.mini_batch_size:", args.mini_batch_size) # 8
            inputs = [x[a:b] for x in input_data]
            break
        break
    
    edge_index_orig, edge_type_orig = inputs[-2:]
    
    print("concept_ids size:", inputs[4].size()) # [8, 5, 200]
    
    #_inputs = [x.reshape(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs[:4]] + [x.reshape(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs[4:-2]] + [sum(x,[]) for x in inputs[-2:]]
    _inputs = [x.reshape(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs[4:-2]] + [sum(x,[]) for x in inputs[-2:]]

    #*lm_inputs, concept_ids, node_type_ids, node_scores, adj_lengths, special_nodes_mask, edge_index, edge_type = _inputs
    concept_ids, node_type_ids, node_scores, adj_lengths, special_nodes_mask, edge_index, edge_type = _inputs

    node_scores = torch.zeros_like(node_scores)
    edge_index, edge_type = batch_graph(edge_index, edge_type, concept_ids.size(1))
    adj = (edge_index.to(node_type_ids.device), edge_type.to(node_type_ids.device))
    
    n_ntype = 4
    n_etype=38
    
    n_concept=799273
    cpnet_vocab_size = n_concept
    
    p_emb=0.2
    dropout_e = nn.Dropout(p_emb)
    
    k=5
    concept_dim=200
    concept_in_dim=1024
    pretrained_concept_emb=None
    freeze_ent_emb=True
    if k >= 0:
        concept_emb = layers.CustomizedEmbedding(concept_num=n_concept, concept_out_dim=concept_dim, use_contextualized=False, concept_in_dim=concept_in_dim, pretrained_concept_emb=pretrained_concept_emb, freeze_ent_emb=freeze_ent_emb)
    
    emb_data=None
    concept_ids[concept_ids == 0] = cpnet_vocab_size + 2
    gnn_input = concept_emb(concept_ids - 1, emb_data).to(node_type_ids.device)
    gnn_input[:, 0] = 0
    gnn_input = dropout_e(gnn_input) #(batch_size, n_node, dim_node)
    
    
    #Normalize node sore (use norm from Z)
    _mask = (torch.arange(node_scores.size(1), device=node_scores.device) < adj_lengths.unsqueeze(1)).float() #0 means masked out #[batch_size, n_node]
    node_scores = -node_scores
    node_scores = node_scores - node_scores[:, 0:1, :] #[batch_size, n_node, 1]
    node_scores = node_scores.squeeze(2) #[batch_size, n_node]
    node_scores = node_scores * _mask
    mean_norm  = (torch.abs(node_scores)).sum(dim=1) / adj_lengths  #[batch_size, ]
    node_scores = node_scores / (mean_norm.unsqueeze(1) + 1e-05) #[batch_size, n_node]
    node_scores = node_scores.unsqueeze(2) #[batch_size, n_node, 1]
    
    
    node_type = node_type_ids
    node_score = node_scores
    
    X = gnn_input
    _X = X.view(-1, X.size(2)).contiguous()
    
    edge_index, edge_type = adj
    
    _node_type = node_type.view(-1).contiguous()
    
    # GNN inputs
    _batch_size, _n_nodes = node_type.size()
    

    
    emb_node_type = nn.Linear(n_ntype, concept_dim // 2)
    
    
    basis_f = 'sin' #['id', 'linact', 'sin', 'none']
    if basis_f in ['id']:
        emb_score = nn.Linear(1, concept_dim // 2)
    elif basis_f in ['linact']:
        B_lin = nn.Linear(1, concept_dim // 2)
        emb_score = nn.Linear(concept_dim // 2, concept_dim // 2)
    elif basis_f in ['sin']:
        emb_score = nn.Linear(concept_dim // 2, concept_dim // 2)

    activation = layers.GELU()
    
    #Embed type
    T = make_one_hot(node_type.view(-1).contiguous(), n_ntype).view(_batch_size, _n_nodes, n_ntype)
    node_type_emb = activation(emb_node_type(T)) #[batch_size, n_node, dim/2]
    
    #Embed score
    if basis_f == 'sin':
        js = torch.arange(concept_dim//2).unsqueeze(0).unsqueeze(0).float().to(node_type.device) #[1,1,dim/2]
        js = torch.pow(1.1, js) #[1,1,dim/2]
        B = torch.sin(js * node_score) #[batch_size, n_node, dim/2]
        node_score_emb = activation(emb_score(B)) #[batch_size, n_node, dim/2]
    elif basis_f == 'id':
        B = node_score
        node_score_emb = activation(emb_score(B)) #[batch_size, n_node, dim/2]
    elif basis_f == 'linact':
        B = activation(B_lin(node_score)) #[batch_size, n_node, dim/2]
        node_score_emb = activation(emb_score(B)) #[batch_size, n_node, dim/2]

    
    _node_feature_extra = torch.cat([node_type_emb, node_score_emb], dim=2).view(_node_type.size(0), -1).contiguous() #[`total_n_nodes`, dim]

    
    return _X, edge_index, edge_type, _node_type, _node_feature_extra

def check_outputs(output):
    print(output)

def test_GATConvE(args):

    n_ntype=4
    n_etype=38
    emb_dim=200
    edge_encoder = torch.nn.Sequential(torch.nn.Linear(n_etype + 1 + n_ntype * 2, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU(), torch.nn.Linear(emb_dim, emb_dim))
    
    device="cpu"
    model = GATConvE(emb_dim, n_ntype, n_etype, edge_encoder).to(device)
    inputs = get_inputs(args)
    outputs = model(*inputs)
    print("outputs size:", outputs.size()) # [8000, 200]
    check_outputs(outputs)


if __name__ == '__main__':
    __spec__ = None

    parser = parser_utils.get_parser()
    args, _ = parser.parse_known_args()

    # General
    parser.add_argument('--mode', default='train', choices=['train', 'eval'], help='run training or evaluation')
    parser.add_argument('--save_dir', default=f'./saved_models/greaselm/', help='model output directory')
    parser.add_argument('--save_model', default=True, type=utils.bool_flag, help="Whether to save model checkpoints or not.")
    parser.add_argument('--load_model_path', default=None, help="The model checkpoint to load in the evaluation mode.")
    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='show this help message and exit')
    parser.add_argument("--run_name", required=True, type=str, help="The name of this experiment run.")
    parser.add_argument("--resume_checkpoint", default=None, type=str,
                        help="The checkpoint to resume training from.")
    parser.add_argument('--use_wandb', default=False, type=utils.bool_flag, help="Whether to use wandb or not.")
    parser.add_argument("--resume_id", default=None, type=str, help="The wandb run id to resume if `resume_checkpoint` is not None or 'None'.")

    # Data
    parser.add_argument('--train_adj', default=f'{args.data_dir}/{args.dataset}/graph/train.graph.adj.pk', help="The path to the retrieved KG subgraphs of the training set.")
    parser.add_argument('--dev_adj', default=f'{args.data_dir}/{args.dataset}/graph/dev.graph.adj.pk', help="The path to the retrieved KG subgraphs of the dev set.")
    parser.add_argument('--test_adj', default=f'{args.data_dir}/{args.dataset}/graph/test.graph.adj.pk', help="The path to the retrieved KG subgraphs of the test set.")
    parser.add_argument('--max_node_num', default=200, type=int, help="Max number of nodes / the threshold used to prune nodes.")
    parser.add_argument('--subsample', default=1.0, type=float, help="The ratio to subsample the training set.")
    parser.add_argument('--n_train', default=-1, type=int, help="Number of training examples to use. Setting it to -1 means using the `subsample` argument to determine the training set size instead; otherwise it will override the `subsample` argument.")

    # Model architecture
    parser.add_argument('-k', '--k', default=5, type=int, help='The number of GreaseLM layers')
    parser.add_argument('--att_head_num', default=2, type=int, help='number of attention heads of the final graph nodes\' pooling')
    parser.add_argument('--gnn_dim', default=100, type=int, help='dimension of the GNN layers')
    parser.add_argument('--fc_dim', default=200, type=int, help='number of FC hidden units (except for the MInt operators)')
    parser.add_argument('--fc_layer_num', default=0, type=int, help='number of hidden layers of the final MLP')
    parser.add_argument('--freeze_ent_emb', default=True, type=utils.bool_flag, nargs='?', const=True, help='Whether to freeze the entity embedding layer.')
    parser.add_argument('--ie_dim', default=200, type=int, help='number of the hidden units of the MInt operator.')
    parser.add_argument('--info_exchange', default=True, choices=[True, False, "every-other-layer"], type=utils.bool_str_flag, help="Whether we have the MInt operator in every GreaseLM layer or every other GreaseLM layer or not at all.")
    parser.add_argument('--ie_layer_num', default=1, type=int, help='number of hidden layers in the MInt operator')
    parser.add_argument("--sep_ie_layers", default=False, type=utils.bool_flag, help="Whether to share parameters across the MInt ops across differernt GreaseLM layers or not. Setting it to `False` means sharing.")
    parser.add_argument('--random_ent_emb', default=False, type=utils.bool_flag, nargs='?', const=True, help='Whether to use randomly initialized learnable entity embeddings or not.')
    parser.add_argument("--cxt_node_connects_all", default=False, type=utils.bool_flag, help="Whether to connect the interaction node to all the retrieved KG nodes or only the linked nodes.")


    # Regularization
    parser.add_argument('--dropouti', type=float, default=0.2, help='dropout for embedding layer')
    parser.add_argument('--dropoutg', type=float, default=0.2, help='dropout for GNN layers')
    parser.add_argument('--dropoutf', type=float, default=0.2, help='dropout for fully-connected layers')

    # Optimization
    parser.add_argument('-dlr', '--decoder_lr', default=DECODER_DEFAULT_LR[args.dataset], type=float, help='Learning rate of parameters not in LM')
    parser.add_argument('-mbs', '--mini_batch_size', default=1, type=int)
    parser.add_argument('-ebs', '--eval_batch_size', default=2, type=int)
    parser.add_argument('--unfreeze_epoch', default=4, type=int, help="Number of the first few epochs in which LM’s parameters are kept frozen.")
    parser.add_argument('--refreeze_epoch', default=10000, type=int)
    parser.add_argument('--init_range', default=0.02, type=float, help='stddev when initializing with normal distribution')

    args = parser.parse_args()
    test_GATConvE(args)
    