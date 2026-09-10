"""Static original CORE90 profile with explicit current protocol overrides.

This module builds argv only. It never executes training or contacts N607.
"""
import argparse
import json
from pathlib import Path


PROFILE_PATH = Path(__file__).resolve().parents[1] / 'configs' / 'core90_evidence_h0_h5.json'


def load_profile():
    return json.loads(PROFILE_PATH.read_text(encoding='utf-8'))


def core90_arguments(dataset, contract, output, variant='H0', seed=392002,
                     device='cuda:0', *, source_rxs, source_days, target_rxs,
                     target_days):
    """Build fully parsed argv; physical dataset selectors are never inferred.

    contract is a path to the actual source/target data contract, also required
    for H0 so the baseline has the same input provenance as H1--H5.
    """
    from SSDG.train_ssdg import build_arg_parser
    profile = load_profile()
    if variant not in profile['variant_configs']:
        raise ValueError('variant must be H0--H5')
    required = dict(dataset=dataset,contract=contract,output=output,
                    source_rxs=source_rxs,source_days=source_days,
                    target_rxs=target_rxs,target_days=target_days)
    if any(not str(value).strip() for value in required.values()):
        raise ValueError('dataset, contract, output and physical selectors are required')
    values = dict(profile['historical_parser_defaults'])
    values.update(profile['later_addon_overrides'])
    values.update(profile['original_command_arguments'])
    values.update(profile['intentional_overrides'])
    replacements = dict(dataset=str(dataset),output=str(output),seed=str(seed),
                        variant=variant,device=str(device))
    values = {key:value.format(**replacements) if isinstance(value,str) else value
              for key,value in values.items()}
    values.update(wisig_train_rxs=source_rxs,wisig_train_days=source_days,
                  wisig_test_rxs=target_rxs,wisig_test_days=target_days,
                  evidence_data_contract=str(contract))
    config = profile['variant_configs'][variant]
    values['evidence_config'] = '' if config is None else json.dumps(config,sort_keys=True)
    parser = build_arg_parser()
    actions = {option:action for action in parser._actions for option in action.option_strings}
    argv = []
    for key,value in values.items():
        option = '--'+key
        if option not in actions:
            raise ValueError(f'historical CORE90 argument is unsupported: {option}')
        action = actions[option]
        if isinstance(action,(argparse._StoreTrueAction,argparse._StoreFalseAction)):
            if bool(value) == action.const:
                argv.append(option)
            elif bool(value) != action.default:
                raise ValueError(f'cannot express historical boolean value for {option}')
        elif value is None:
            if action.default is not None:
                raise ValueError(f'cannot preserve historical None for {option}')
        else:
            argv.extend((option,str(value).lower() if isinstance(value,bool) else str(value)))
    # Unknown and invalid arguments are fatal, never silently dropped.
    parser.parse_args(argv)
    return argv
