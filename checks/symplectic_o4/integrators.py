"""Experimental fixed-step integrators; no production dispatch changes.

VJP contraction is J_a.T @ a (equal masses only). For a conservative
force its Jacobian is symmetric, giving the FROST J_a @ a contraction.
FMM approximation/interaction-list changes can spoil this equivalence.
"""
import jax


def force_gradient(acceleration, x):
    a, pullback = jax.vjp(acceleration, x)
    return a, pullback(a)[0]


def kdk(x, v, a, h, acceleration):
    v = v + h * a / 2
    x = x + h * v
    a = acceleration(x)
    return x, v + h * a / 2, a


def s4g(x, v, a, h, acceleration):
    v = v + h * a / 6
    x = x + h * v / 2
    amid, contraction = force_gradient(acceleration, x)
    v = v + (2 * h / 3) * (amid + h**2 * contraction / 24)
    x = x + h * v / 2
    a = acceleration(x)
    return x, v + h * a / 6, a


def fmm_s4g(p, h, cfg):
    """Experimental self-gravity step; equal masses, no external/time-dependent field.

    Input p.loc must contain the endpoint force from the previous step.
    Callers enforce equal masses; this benchmark helper is not production API.
    """
    from dataclasses import replace
    from jzfmm.time_integration import force_and_potential
    if cfg.external_potential is not None:
        raise ValueError('Only autonomous self-gravity is supported here')
    assert p.loc is not None
    acceleration=lambda x:force_and_potential(replace(p,pos=x,loc=None),cfg).force()
    v=p.vel+h*p.loc.force()/6
    x=p.pos+h*v/2
    a,g=force_gradient(acceleration,x)
    v=v+(2*h/3)*(a+h*h*g/24)
    x=x+h*v/2
    p=replace(p,pos=x)
    loc=force_and_potential(p,cfg)
    return replace(p,vel=v+h*loc.force()/6,loc=loc)
