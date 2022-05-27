'''
Created on Apr. 4, 2022

@author: yfang
'''
import torch
import random
import numpy as np
from utils import data_utils
from utils import parser_utils
from utils import utils
import argparse
from tqdm import tqdm, trange
from model.utils import get_model, TaskType
from transformers import (
    AutoConfig,
    AutoTokenizer,
)


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



    
def test_PT(args):
    
    
    print("args: {}".format(args))
    
    kg = "cpnet"
    if args.dataset == "medqa_usmle":
        kg = "ddb"
    
    devices = get_devices(args.cuda)
    
    dataset = load_data(args, devices, kg)
    train_dataloader = dataset.train()
    
    
    
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
    _inputs = [x.reshape(x.size(0) , x.size(1), *x.size()[2:]) for x in inputs[:4]] + [x.reshape(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs[4:-2]] + [sum(x,[]) for x in inputs[-2:]]
    

    *lm_inputs, concept_ids, node_type_ids, node_scores, adj_lengths, special_nodes_mask, edge_index, edge_type = _inputs
    
    
    input_ids, attention_mask, token_type_ids, output_mask = lm_inputs    
    inputs = input_ids, token_type_ids, attention_mask
    print("input_ids size:", input_ids.size())
    print("input_ids:", list(input_ids.numpy())[0])
    print("token_type_ids size:", token_type_ids.size())
    print("token_type_ids:", list(token_type_ids.numpy())[0])
    print("attention_mask size:", attention_mask.size())
    print("attention_mask:", list(attention_mask.numpy())[0])
    print("output_mask size:", output_mask.size())
    print("output_mask:", list(output_mask.numpy())[0])
    
    labels = dataset.train_labels.numpy().astype(np.int32)
    print("dataset.train_labels:", labels)
    print("len(dataset.train_labels):", len(labels))
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        #num_labels=len(labels),
        #label2id={l: i for i, l in enumerate(labels)},
        #id2label=[l for l in labels],
        finetuning_task=args.dataset,  
        revision=args.model_revision,
    )

    device="cpu"
    model = get_model(args, TaskType.MULTIPLE_CHOICE, config, fix_bert=True)
    #model = get_model(args, TaskType.SEQUENCE_CLASSIFICATION, config)
    #model.resize_token_embeddings(len(dataset.tokenizer))
    print("model:", model)
    
    outputs = model(*inputs)
    print("outputs.logits size:", outputs.logits.size()) 
    print("outputs.logits:", outputs.logits)
    #print("outputs.hidden_states size:", outputs.hidden_states.size())
    


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


    # Additional Model Arguments
    parser.add_argument("--model_name_or_path", default=f"{args.encoder}", type=str, help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--config_name", default=None, type=str, help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default=None, type=str, help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default=None, type=str, help="Where do you want to store the pretrained models downloaded from huggingface.co")
    parser.add_argument("--use_fast_tokenizer", default=True, type=bool, help="Whether to use one of the fast tokenizer (backed by the tokenizers library) or not.")
    parser.add_argument("--model_revision", default="main", type=str, help="The specific model version to use (can be a branch name, tag name or commit id).")
    parser.add_argument("--use_auth_token", default=False, type=bool, help="Will use the token generated when running `transformers-cli login` (necessary to use this script with private models).")
    parser.add_argument("--prefix", default=True, type=bool, help="Will use P-tuning v2 during training")
    parser.add_argument("--prompt", default=False, type=bool, help="Will use prompt tuning during training")
    parser.add_argument("--pre_seq_len", default=128, type=int, help="The length of prompt")
    parser.add_argument("--prefix_projection", default=False, type=bool, help="Apply a two-layer MLP head over the prefix embeddings")
    parser.add_argument("--prefix_hidden_size", default=512, type=int, help="The hidden size of the MLP projection head in Prefix Encoder if prefix projection is used")
    parser.add_argument("--hidden_dropout_prob", default=0.1, type=float, help="The dropout probability used in the models")


    args = parser.parse_args()
    test_PT(args)    