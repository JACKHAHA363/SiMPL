import torch
from collections import OrderedDict
import argparse
from spirl.rl.policies.cl_model_policies import ClModelPolicy
from spirl.configs.hrl.kitchen.spirl.conf import *
from spirl.models.closed_loop_spirl_mdl import ClSPiRLMdl
from spirl.rl.policies.cl_model_policies import ClModelPolicy


parser = argparse.ArgumentParser()
parser.add_argument('--spirl_weight', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()

# Get spirl low policy
ll_model_params.cond_decode = True

# create LL closed-loop policy
ll_policy_params = AttrDict(
    policy_model=ClSPiRLMdl,
    policy_model_params=ll_model_params,
    policy_model_checkpoint=args.spirl_weight)

spirl_low_policy = ClModelPolicy(ll_policy_params)
state_dict = spirl_low_policy.net.decoder.state_dict()
spirl_low_policy_state_dict = OrderedDict()
spirl_low_policy_state_dict['log_sigma'] = spirl_low_policy._log_sigma
for k, v in state_dict.items():
    names = k.split('.')
    if names[0] == 'input':  # input.linear->0
        spirl_low_policy_state_dict[f'net.net.0.{names[-1]}'] = v
    elif names[0].startswith('pyramid'):  # pyramid-i.linear->3*i+1, pyramid-i.linear->3*i+2
        pyramid_n = int(names[0][-1])
        is_norm = (names[1] == 'norm')
        layer_n = 3*pyramid_n + int(is_norm) + 2
        spirl_low_policy_state_dict[f'net.net.{layer_n}.{names[-1]}'] = v
    elif names[0] == 'head':  # input.linear->{last_n+2}
        spirl_low_policy_state_dict[f'net.net.{layer_n+2}.{names[-1]}'] = v
        
state_dict = spirl_low_policy.net.p[0].state_dict()
spirl_prior_policy_state_dict = OrderedDict()
for k, v in state_dict.items():
    names = k.split('.')
    if names[0] == 'input':  # input.linear->0
        spirl_prior_policy_state_dict[f'net.net.0.{names[-1]}'] = v
    elif names[0].startswith('pyramid'):  # pyramid-i.linear->3*i+2, pyramid-i.linear->3*i+3
        pyramid_n = int(names[0][-1])
        is_norm = (names[1] == 'norm')
        layer_n = 3*pyramid_n + int(is_norm) + 2
        spirl_prior_policy_state_dict[f'net.net.{layer_n}.{names[-1]}'] = v
    elif names[0] == 'head':  # input.linear->{last_n+2}
        spirl_prior_policy_state_dict[f'net.net.{layer_n+2}.{names[-1]}'] = v        

torch.save({
    'spirl_low_policy': spirl_low_policy_state_dict,
    'spirl_prior_policy': spirl_prior_policy_state_dict,
    'horizon': 10, 'z_dim': 10
}, args.output)