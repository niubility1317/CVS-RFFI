"""Focused source-audit release checks, synthetic only and no optimizer of E."""
import json
from .geometry_checks import run as geometry
from . import receiver_checks,physics_checks,mechanism_checks


def run():
    from . import design as d
    from .audit_control import RUN
    assert RUN==d.AUDIT_RUN and len(d.rows())==12
    result={'geometry':geometry()}
    result['receiver']=receiver_checks.check()
    result['physics']=physics_checks.run()
    result['objectives']=mechanism_checks.objective_checks()
    result['actions']=mechanism_checks.action_checks()
    return dict(status='PASS',target_access=False,identity_training_launched=False,checks=result)


if __name__=='__main__':print(json.dumps(run()))
