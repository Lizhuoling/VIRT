import os
import yaml
import pdb
import copy

def load_yaml_with_base(filename):
    with open(filename, 'r') as file:
        config = yaml.safe_load(file)

    if '__BASE__' in config:
        base_config = load_yaml_with_base(os.path.join('configs', config['__BASE__']))
        del config['__BASE__']
        config = merge_dicts(base_config, config)

    return config

def merge_dicts(A, B):
    for key in B:
        if key in A and isinstance(A[key], dict) and isinstance(B[key], dict):
            A[key] = merge_dicts(A[key], B[key])
        else:
            A[key] = B[key]
    return A