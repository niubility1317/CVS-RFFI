"""Opt-in original-LEO scenario mapping and genuine CE-only concat forward."""
import torch

ORIGINAL = ('clean', 'clear_leo', 'low_elev_leo', 'rain_leo')
WEAK = ('clean', 'leo_clear_weak', 'leo_low_elev_weak', 'leo_rain_weak')


def original_scenario(name):
    return dict(zip(WEAK[1:], ORIGINAL[1:])).get(name, name)


def validate_scenarios(scenarios):
    result = tuple(scenarios)
    if result not in (ORIGINAL, WEAK):
        raise ValueError('Expected a registered clean plus three original/weak LEO scenes')
    return result


class FusedCleanSatelliteForward:
    """One model invocation; retain clean-only auxiliaries and cached satellite CE.

    Both views share the same forward/batch statistics. Outputs passed back to
    the ordinary supervised objectives contain clean rows only. The satellite
    output graph is used exclusively by the existing auxiliary CE branch.
    """
    def __init__(self, model, satellite):
        self.model = model
        self.satellite = satellite
        self.satellite_output = None

    def __call__(self, clean, **kwargs):
        if clean.shape != self.satellite.shape:
            raise ValueError('Clean and satellite batch shapes must match')
        count = clean.shape[0]
        if kwargs.get('update_crra_support', False) or kwargs.get('crra_support_mask') is not None:
            raise ValueError('This CE-only concat path does not enable CRRA support updates')
        for key in ('y_tx', 'domain_labels'):
            value = kwargs.get(key)
            if value is not None:
                kwargs[key] = torch.cat((value, value), dim=0)
        output = self.model(torch.cat((clean, self.satellite), dim=0), **kwargs)
        def split(value, begin):
            if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == 2 * count:
                return value[begin:begin+count]
            if isinstance(value, dict):
                return {key: split(item, begin) for key, item in value.items()}
            if isinstance(value, tuple):
                return tuple(split(item, begin) for item in value)
            if isinstance(value, list):
                return [split(item, begin) for item in value]
            return value
        self.satellite_output = split(output, count)
        return split(output, 0)
