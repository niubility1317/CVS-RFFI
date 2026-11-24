"""Separate observed-noise weighting and discrepancy stopping with a known source action."""

import numpy as np
from scipy.sparse.linalg import LinearOperator

from .packet_physics import chirp_samples, pack_windows
from .physical_inverse import full_action, real_operator, residual_actions, solve_observations
from .registered_inverse import projected_operator, sampling_matrix


RATIOS = (1., .75)
MAX_ITERATIONS = 30


def real_iq(value):
    return np.concatenate((value.real,value.imag))


def complex_iq(value):
    a,b = np.split(value,2)
    return a+1j*b


def weight_operator(operator, weight):
    return LinearOperator(operator.shape, matvec=lambda x:weight*operator.matvec(x),
                          rmatvec=lambda y:operator.rmatvec(weight*y), dtype=np.float64)


def discrepancy_solutions(operator, observation, complex_variance):
    """CGLS, preserving the first nonzero iterates meeting each energy budget."""
    rhs = real_iq(observation)
    residual = rhs.copy()
    x = np.zeros(operator.shape[1],dtype=np.float64)
    previous = operator.rmatvec(residual)
    direction = previous.copy()
    gamma = previous@previous
    total = [observation.size*complex_variance*r*r for r in RATIOS]
    saved = [None]*len(RATIOS)
    step_used = [0]*len(RATIOS)
    for i,budget in enumerate(total):
        if residual@residual <= budget:
            saved[i] = x.copy()
    if all(val is not None for val in saved):
        return tuple(complex_iq(val) for val in saved),step_used
    executed = 0
    for iteration in range(1,MAX_ITERATIONS+1):
        u = operator.matvec(direction)
        energy = u@u
        if energy <= np.finfo(float).tiny:
            break
        executed = iteration
        alpha = gamma/energy
        x += alpha*direction
        residual -= alpha*u
        power = residual@residual
        for i,limit in enumerate(total):
            if saved[i] is None and power <= limit:
                saved[i],step_used[i] = x.copy(),iteration
        if all(s is not None for s in saved):
            break
        new = operator.rmatvec(residual)
        length = new@new
        if length <= 1e-26:
            break
        direction = new + length/gamma*direction
        gamma = length
    if not np.isfinite(x).all():
        raise FloatingPointError('Known-action CGLS produced nonfinite source IQ')
    return (tuple(complex_iq(x if val is None else val) for val in saved),
            [(-max(1,executed) if val is None else step) for val,step in zip(saved,step_used,strict=True)])


def solve_packet(operator, raw, clean, sigma):
    """Keep all source truth outside the competing raw-observation inverses."""
    level = float(np.linalg.norm(sigma)/np.sqrt(sigma.size))
    if not np.isfinite(level) or level<=1e-10:
        raise ValueError('Need positive received-noise estimates per chirp')
    weights = level/np.clip(sigma,.25*level,4.*level)
    ordered = np.repeat(weights,raw.size//len(sigma))
    weighted = weight_operator(operator,np.tile(ordered,2))
    damping = np.sqrt(3)*level
    known80 = clean+.2*(raw-clean)
    base,known_inverse = solve_observations(operator,(raw,known80),damping)
    weighted_raw = raw*ordered
    weighted_damped, = solve_observations(weighted,(weighted_raw,),damping)
    equal,steps_equal = discrepancy_solutions(operator,raw,level**2)
    weighted_variance = float(np.mean((sigma*weights)**2))
    unequal,steps_weighted = discrepancy_solutions(weighted,weighted_raw,weighted_variance)
    return (base,weighted_damped,*equal,*unequal,known_inverse),steps_equal+steps_weighted


def inverse_full(packet,scene,rx,pid,registered,noise_proxy):
    action,source,received,signal,_,gain,offset,error = full_action(packet,scene,rx,pid)
    sample = sampling_matrix(len(received),offset)
    native_chirps = chirp_samples(received,offset)
    if not np.array_equal(registered,pack_windows(native_chirps)):
        raise ValueError('Full exact action needs the registered source reception')
    observed = native_chirps.astype(np.complex128).ravel()/gain
    noiseless = sample@signal
    mismatch = np.linalg.norm(sample@action(source)-noiseless)/max(np.linalg.norm(noiseless),1e-12)
    if mismatch>2e-9:
        raise ValueError('Full projected action does not reproduce the six registered windows')
    unit = np.sqrt(np.mean(np.abs(native_chirps)**2))/gain
    solutions,steps = solve_packet(projected_operator(action,sample),observed,noiseless,noise_proxy*unit)
    outputs = [pack_windows(chirp_samples(value)) if np.any(value) else registered for value in solutions]
    return np.stack(outputs),steps,max(float(mismatch),error)


def inverse_residual(packet,scene,rx,pid,registered,noise_proxy):
    records = list(residual_actions(packet,scene,rx,pid))
    captures = np.stack([row[1] for row in records])
    if not np.array_equal(registered,pack_windows(captures)):
        raise ValueError('Residual exact action needs the registered source reception')
    operators = [row[0] for row in records]
    sizes = [o.propagation.shape for o in operators]
    inputs = np.cumsum([k[1] for k in sizes])[:-1]
    outputs = np.cumsum([k[0] for k in sizes])[:-1]
    action = real_operator(
        lambda x:np.concatenate([o(z) for o,z in zip(operators,np.split(x,inputs),strict=True)]),
        lambda y:np.concatenate([o.adjoint_iq(z) for o,z in zip(operators,np.split(y,outputs),strict=True)]),
        (sum(k[0] for k in sizes),sum(k[1] for k in sizes)))
    gains = np.array([row[4] for row in records])
    receive = (captures.astype(np.complex128)/gains[:,None]).ravel()
    signal = np.concatenate([row[2] for row in records])
    unit = np.sqrt(np.mean(np.abs(captures)**2))/gains
    candidates,steps = solve_packet(action,receive,signal,noise_proxy*unit)
    packed = [pack_windows(np.stack([p[32:288] for p in np.split(x,inputs)])) if np.any(x)
              else registered for x in candidates]
    return np.stack(packed),steps,max(k[-1] for k in records)


def inverse_row(route, scene, row):
    packet,rx,pid,registered,proxy = row
    solve = inverse_full if route=='full' else inverse_residual
    return solve(packet,scene,rx,pid,registered,proxy)
