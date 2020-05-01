import argparse
import PIL
import os
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from mobilenet import mobilenet
from torch.autograd import Variable
from datetime import datetime
from ast import literal_eval
import json
from torchvision.utils import save_image
from tqdm import tqdm
import nemo
import warnings
import math
import copy
import collections 

# -a mobilenet \
#   --mobilenet_width 1.0 \
#   --mobilenet_input 128 \
#   --dataset imagenet \
#   --weight_bits 8 \
#   --activ_bits 8 \
#   --gpus 0 \
#   -j 8 \
#   --epochs 12 \
#   -b 128 \
#   --quantize \
#   --terminal \
#   --resume checkpoint/mobilenet_1.0_128_best.pth

# filter out ImageNet EXIF warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
warnings.filterwarnings("ignore", "Metadata Warning", UserWarning)

model_names = ['mobilenet',]

parser = argparse.ArgumentParser(description='PyTorch ConvNet Training')

parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results',
                    help='results dir')
parser.add_argument('--save', metavar='SAVE', default='',
                    help='saved folder')
parser.add_argument('--dataset', metavar='DATASET', default='imagenet',
                    help='dataset name or folder')
parser.add_argument('--model', '-a', metavar='MODEL', default='mobilenet',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: alexnet)')
parser.add_argument('--input_size', type=int, default=None,
                    help='image input size')
parser.add_argument('--model_config', default='',
                    help='additional architecture configuration')
parser.add_argument('--type', default='torch.FloatTensor',
                    help='type of tensor - e.g torch.cuda.HalfTensor')
parser.add_argument('--gpus', default='',
                    help='gpus used for training - e.g 0,1,3')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--optimizer', default='SGD', type=str, metavar='OPT',
                    help='optimizer function used')
parser.add_argument('--lr', '--learning_rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default="mobilenet_1.0_128_best.pth", type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', action='store_true',
                    help='run model on validation set')
parser.add_argument('--save_check', action='store_true',
                    help='saving the checkpoint')
parser.add_argument('--terminal', action='store_true')
# quantization parameters
parser.add_argument('--quantize', default=True, action='store_true',
                    help='quantize the network')
parser.add_argument('--type_quant', default=None,
                    help='Type of binarization process')
parser.add_argument('--weight_bits', default=8,
                    help='Number of bits for the weights')
parser.add_argument('--activ_bits', default=8,
                    help='Number of bits for the activations')

parser.add_argument('--initial_folding', default=False, action='store_true',
                    help='Fold BNs into Linear layers before training')
parser.add_argument('--initial_equalization', default=False, action='store_true',
                    help='Perform Linear layer weight equalization before training')
parser.add_argument('--quant_add_config', default='', type=str, 
                    help='Additional config of per-layer quantization')

# mobilenet params
parser.add_argument('--mobilenet_width', default=1.0, type=float,
                    help='Mobilenet Width Muliplier')
parser.add_argument('--mobilenet_input', default=128, type=int,
                    help='Mobilenet input resolution ')

# mixed-precision params
parser.add_argument('--mem_constraint', default='', type=str,
                    help='Memory constraints for automatic bitwidth quantization')
parser.add_argument('--mixed_prec_quant', default='MixPL', type=str, 
                    help='Type of quantization for mixed-precision low bitwidth: MixPL | MixPC')


def main():
    global args, best_prec1
    best_prec1 = 0
    args = parser.parse_args()
    
    weight_bits = int(args.weight_bits)
    activ_bits = int(args.activ_bits)

    print("run arguments: %s", args)

    args.gpus = None

    # create model
    print("creating model %s", args.model)
    model_config = {'input_size': args.input_size, 'dataset': args.dataset, 'num_classes': 1000, \
                    'width_mult': float(args.mobilenet_width), 'input_dim': float(args.mobilenet_input) }

    if args.model_config is not '':
        model_config = dict(model_config, **literal_eval(args.model_config))

    model = mobilenet(**model_config).to('cuda')
    print("created model with configuration: %s", model_config)
    print(model)


    num_parameters = sum([l.nelement() for l in model.parameters()])
    print("number of parameters: %d", num_parameters)

    mobilenet_width = float(args.mobilenet_width)
    mobilenet_input = int(args.mobilenet_input) 

    # transform the model in a NEMO FakeQuantized representation
    model = nemo.transform.quantize_pact(model, dummy_input=torch.randn((1,3,mobilenet_input,mobilenet_input)).to('cuda'))

    checkpoint_file = args.resume
    if os.path.isfile(checkpoint_file):
        print("loading checkpoint '%s'", args.resume)
        checkpoint_loaded = torch.load(checkpoint_file) #, map_location=torch.device('cuda'))
        checkpoint = checkpoint_loaded['state_dict']
        model.load_state_dict(checkpoint, strict=True)
        prec_dict = checkpoint_loaded.get('precision')
    else:
        print("no checkpoint found at '%s'", args.resume)
        import sys; sys.exit(1)

    print("[NEMO] Not calibrating model, as it is pretrained")
    model.change_precision(bits=1, min_prec_dict=prec_dict)

    inputs = torch.load("input_fq.pth")['in'] # , map_location=torch.device('cuda'))['in']
    bin_fq, bout_fq, _ = nemo.utils.get_intermediate_activations(model, forward, model, inputs)

    input_bias_dict  = {}# {'model.0.0' : +1.0, 'model.0.1' : +1.0}
    remove_bias_dict = {}#{'model.0.1' : 'model.0.2'}
    input_bias       = 0 #math.floor(1.0 / (2./255)) * (2./255)

    model.qd_stage(eps_in=2./255, int_accurate=False)
    # fix ConstantPad2d
    # model.model[0][0].value = input_bias

    bin_qd, bout_qd, _ = nemo.utils.get_intermediate_activations(model, forward, model, inputs, input_bias=input_bias)

    diff = collections.OrderedDict()
    for k in bout_fq.keys():
        diff[k] = (bout_fq[k] - bout_qd[k]).to('cpu').abs()
    print(torch.get_default_dtype())

    for i in range(0,26):
        for j in range(3,4):
            k  = 'model.%d.%d' % (i,j)
            kn = 'model.%d.%d' % (i if j<3 else i+1, j+1 if j<3 else 0)
            eps = model.get_eps_at(kn, eps_in=2./255)[0]
            print("%s:" % k)
            idx = diff[k]>eps
            n = idx.sum()
            t = (diff[k]>-1e9).sum()
            max_eps = torch.ceil(diff[k].max() / model.get_eps_at('model.%d.0' % (i+1), 2./255)[0]).item()
            mean_eps = torch.ceil(diff[k][idx].mean() / model.get_eps_at('model.%d.0' % (i+1), 2./255)[0]).item()
            assert(max_eps < 1)
            try:
                print("  max:   %.3f (%d eps)" % (diff[k].max().item(), max_eps))
                print("  mean:  %.3f (%d eps) (only diff. elements)" % (diff[k][idx].mean().item(), mean_eps))
                print("  #diff: %d/%d (%.1f%%)" % (n, t, float(n)/float(t)*100)) 
            except ValueError:
                print("  #diff: 0/%d (0%%)" % (t,)) 

    # model.id_stage()
    # # fix ConstantPad2d
    # # model.model[0][0].value = input_bias / (2./255)

    # ids = model.state_dict()
    # bin_id, bout_id, _ = nemo.utils.get_intermediate_activations(model, validate, val_loader, model, criterion, 0, None, input_bias=input_bias, shorten=1, eps_in=2./255) 

    # diff = collections.OrderedDict()
    # for i in range(0,26):
    #     for j in range(3,4):
    #         k  = 'model.%d.%d' % (i,j)
    #         kn = 'model.%d.%d' % (i if j<3 else i+1, j+1 if j<3 else 0)
    #         eps = model.get_eps_at(kn, eps_in=2./255)[0]
    #         diff[k] = (bout_id[k]*eps - bout_qd[k]).to('cpu').abs()
    #         print("%s:" % k)
    #         idx = diff[k]>=eps
    #         n = idx.sum()
    #         t = (diff[k]>-1e9).sum()
    #         max_eps  = torch.ceil(diff[k].max() / eps).item()
    #         mean_eps = torch.ceil(diff[k][idx].mean() / eps).item()
    #         try:
    #             print("  max:   %.3f (%d eps)" % (diff[k].max().item(), max_eps))
    #             print("  mean:  %.3f (%d eps) (only diff. elements)" % (diff[k][idx].mean().item(), mean_eps))
    #             print("  #diff: %d/%d (%.1f%%)" % (n, t, float(n)/float(t)*100)) 
    #         except ValueError:
    #             print("  #diff: 0/%d (0%%)" % (t,)) 

def forward(model, inputs, input_bias=0.0, eps_in=None, integer=False):

    model.eval()

    # input quantization
    if eps_in is None:
        scale_factor = 1.
        div_factor   = 1.
    elif not integer:
        scale_factor = 1./eps_in
        div_factor   = 1./eps_in
    else:
        scale_factor = 1./eps_in
        div_factor   = 1.

    # measure data loading time
    with torch.no_grad():
        if eps_in is None:
            input_var = (inputs + input_bias)
        else:
            input_var = (inputs + input_bias) * scale_factor

    # compute output
    output = model(input_var)
    
if __name__ == '__main__':
    main()
