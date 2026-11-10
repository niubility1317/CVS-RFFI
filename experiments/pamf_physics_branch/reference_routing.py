"""Conservative direction correction for a discrete reference-delay change."""
import torch


def stable_reference_directions(directions, raw_delay, perturbed_delay):
    """Retain every sample, but suppress its direction after a route jump."""
    if (directions.ndim != 2 or raw_delay.shape != directions.shape[:1]
            or perturbed_delay.shape != raw_delay.shape
            or raw_delay.device != directions.device or perturbed_delay.device != directions.device
            or not bool(torch.isfinite(directions).all())):
        raise ValueError('Finite directions and aligned delay selections required')
    stable = raw_delay == perturbed_delay
    return directions.detach() * stable[:, None].to(directions.dtype)


def route_stable_physical_directions(student, inputs, seed, measure):
    """Use the original physical probe, masking only jumped reference directions."""
    from .geometry import bounded_echo
    from .objective import echo_parameters

    directions, norms = measure(student, inputs, seed)
    if 'reference' not in directions:
        raise ValueError('The actual reference branch is required')
    with torch.no_grad():
        coefficients, delays = echo_parameters(len(inputs), inputs.device, inputs.dtype, seed)
        augmented = bounded_echo(inputs, coefficients, delays)
        response = student.id_backbone.encoder.response
        original = response.components(inputs)['selected_delay_samples']
        changed = response.components(augmented)['selected_delay_samples']
        reference = stable_reference_directions(directions['reference'], original, changed)
    result = dict(directions, reference=reference)
    student._reference_routing_calls = getattr(student, '_reference_routing_calls', 0) + 1
    student._reference_routing_seen = getattr(student, '_reference_routing_seen', 0) + len(inputs)
    student._reference_routing_changed = (getattr(student, '_reference_routing_changed', 0)
                                          + int((original != changed).sum()))
    return result, norms
