import torch
from collections import OrderedDict
import argparse
from spirl.rl.policies.cl_model_policies import ClModelPolicy
from spirl.configs.hrl.kitchen.spirl.conf import *
from spirl.models.closed_loop_spirl_mdl import ClSPiRLMdl
from spirl.rl.policies.cl_model_policies import ClModelPolicy

from simpl.alg.spirl.spirl_policy import SpirlLowPolicy
from simpl.alg.spirl.spirl_policy import SpirlPriorPolicy

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

clmodelpolicy = ClModelPolicy(ll_policy_params)
state_dict = clmodelpolicy.net.decoder.state_dict()
low_policy_state_dict = OrderedDict()
low_policy_state_dict['log_sigma'] = clmodelpolicy._log_sigma.repeat(clmodelpolicy._hp['policy_model_params']['action_dim'])
for k, v in state_dict.items():
    names = k.split('.')
    if names[0] == 'input':  # input.linear->0
        low_policy_state_dict[f'net.net.0.{names[-1]}'] = v
    elif names[0].startswith('pyramid'):  # pyramid-i.linear->3*i+1, pyramid-i.linear->3*i+2
        pyramid_n = int(names[0][-1])
        is_norm = (names[1] == 'norm')
        layer_n = 3*pyramid_n + int(is_norm) + 2
        low_policy_state_dict[f'net.net.{layer_n}.{names[-1]}'] = v
    elif names[0] == 'head':  # input.linear->{last_n+2}
        low_policy_state_dict[f'net.net.{layer_n+2}.{names[-1]}'] = v

# Initialize New low policy
old_params = clmodelpolicy._hp['policy_model_params']
new_low_policy = SpirlLowPolicy(
    state_dim=old_params['state_dim'],
    z_dim=old_params['nz_vae'],
    action_dim=old_params['action_dim'],
    hidden_dim=old_params['nz_enc'],
    n_hidden=old_params['n_processing_layers'] + 1,
)
msg = new_low_policy.load_state_dict(low_policy_state_dict, strict=False)
print("########### Low policy", msg)


state_dict = clmodelpolicy.net.p[0].state_dict()
prior_policy_state_dict = OrderedDict()
for k, v in state_dict.items():
    names = k.split('.')
    if names[0] == 'input':  # input.linear->0
        prior_policy_state_dict[f'net.net.0.{names[-1]}'] = v
    elif names[0].startswith('pyramid'):  # pyramid-i.linear->3*i+2, pyramid-i.linear->3*i+3
        pyramid_n = int(names[0][-1])
        is_norm = (names[1] == 'norm')
        layer_n = 3*pyramid_n + int(is_norm) + 2
        prior_policy_state_dict[f'net.net.{layer_n}.{names[-1]}'] = v
    elif names[0] == 'head':  # input.linear->{last_n+2}
        prior_policy_state_dict[f'net.net.{layer_n+2}.{names[-1]}'] = v        

# Initialize New prior policy
new_prior_policy = SpirlPriorPolicy(
    state_dim=old_params['state_dim'],
    z_dim=old_params['nz_vae'],
    hidden_dim=old_params['nz_enc'],
    n_hidden=old_params['n_processing_layers'] + 2,
)
msg = new_prior_policy.load_state_dict(prior_policy_state_dict, strict=False)
print("########## Prior policy", msg)


torch.save({
    'spirl_low_policy': new_low_policy,
    'spirl_prior_policy': new_prior_policy,
    'horizon': 10, 'z_dim': 10
}, args.output)