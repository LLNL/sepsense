#!/usr/bin/env python3

"""
This file contains classes used for storing parameters.
The only parameter not accounted for by this framework
is the CUDA_VISIBLE_DEVICES to be used by the experiment.
This should be set at runtime.
"""

import yaml
import collections
import numpy as np
import argparse


TypeDict = {
    'bool': bool,
    'int': int,
    'float': float,
    'str': str,
    'list': list,
    'array': np.array,
}

def parse_cmdline():
    # Parse the few command-line args needed
    parser = argparse.ArgumentParser()
    parser.add_argument('--template_path', type=str, default='./params.yaml')
    parser.add_argument('--param_path', type=str, default='./exp1.yaml')
    parser.add_argument('--test_param_path', type=str, default=None)
    args = parser.parse_args()
    return args

def parse_params(template_path, param_path):
    # Load parameter template from yaml
    with open(template_path, 'r') as fp:
        template_dict = yaml.safe_load(fp)
    # Load experimental parameters from yaml
    with open(param_path, 'r') as fp:
        param_dict = yaml.safe_load(fp)
    # Create named tuple to store all parameters in one object
    Parameters = collections.namedtuple('Parameters', template_dict.keys())
    nt_dict = {}
    for category in template_dict:
        # Create named tuple to store parameters of each category
        CategoryParameters = collections.namedtuple(category, template_dict[category].keys())
        if category in param_dict:
            # Make sure each parameter is of the correct type and is in the allowed choices
            for field in template_dict[category]:
                if field in param_dict[category]:
                    # Get python typename from string
                    field_type = TypeDict[template_dict[category][field]['type']]
                    # Special case for iterable parameters
                    if template_dict[category][field]['type'] != 'array' and type(param_dict[category][field]) == list:
                        param_dict[category][field] = np.arange(*param_dict[category][field])
                    elif type(param_dict[category][field]) == field_type:
                        pass
                    else:
                        try:
                            #print(category, field, param_dict[category][field])
                            param_dict[category][field] = field_type(param_dict[category][field])
                        except:
                            print(category, field)
                            raise Exception('Cannot be converted:')# {} -> {}'.format())
                else:
                    raise Exception('Missing field: <{}>'.format(field))
            cp = CategoryParameters(**param_dict[category]) 
            nt_dict[category] = cp
        else:
            raise Exception('Parameter category <{}> missing from yaml'.format(category))
    # Build the parameter object using the dict of named tuples
    p = Parameters(**nt_dict)

    return p


if __name__=='__main__':        
    #
    args = parse_cmdline()
    p = parse(args.template_path, args.param_path)
    d = p.sensor._asdict()
    d['focal_length'] = 5
    from pprint import pprint
    pprint(d)
    from imsim import sensor_model_degrade_dg as sd
    sd.imageSystem(**d)
