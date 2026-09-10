"""Parameter identity is the ownership key; module names are aliases."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ParameterRole:
    name: str
    aliases: tuple[str, ...]
    role: str
    parameter: object


def parameter_roles(model, *, head_names=("adv_head.",)):
    aliases = {}
    parameters = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)
        parameters[id(parameter)] = parameter
    result = []
    for key, names in aliases.items():
        if any(any(name.startswith(prefix) for prefix in head_names) for name in names):
            role = "adversarial_head"
        elif len(names) > 1 or any(name.startswith(("sinc.", "hf.")) for name in names):
            role = "shared_frontend"
        elif any(name.startswith("dom_head.") for name in names):
            role = "domain_head"
        elif any(name.startswith(("tx_head.", "classifier.", "head.")) for name in names):
            role = "identity_head"
        elif any(name.startswith(("dom_", "domain_")) for name in names):
            role = "domain_branch"
        else:
            role = "encoder_or_auxiliary"
        result.append(ParameterRole(names[0], tuple(names), role, parameters[key]))
    return result


def validate_optimizer_ownership(model, optimizer):
    model_ids = {id(p) for p in model.parameters()}
    seen = set()
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if id(parameter) not in model_ids:
                raise ValueError("Optimizer contains a parameter outside the model")
            if id(parameter) in seen:
                raise ValueError("A shared parameter has more than one optimizer owner")
            seen.add(id(parameter))
    missing = {id(p) for p in model.parameters() if p.requires_grad} - seen
    if missing:
        raise ValueError("Optimizer omits trainable parameters from the coupled field")
    return seen


def role_manifest(model, **kwargs):
    return [{"name": row.name, "aliases": list(row.aliases), "role": row.role,
             "shape": list(row.parameter.shape), "trainable": row.parameter.requires_grad}
            for row in parameter_roles(model, **kwargs)]
