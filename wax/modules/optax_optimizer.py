import haiku as hk
import optax


class OptaxOptimizer(hk.Module):
    """A module which wraps an optax GrandientTransformation.

    Args:
        opt: gradient transformation
        name: name of the module
    """

    def __init__(self, opt: optax.GradientTransformation, name=None):
        super().__init__(name=name)
        self.opt = opt

    def __call__(self, params, grads):
        opt_state = hk.get_state("opt_state", [], init=lambda *_: self.opt.init(params))
        updates, opt_state = self.opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        hk.set_state("opt_state", opt_state)
        return params
