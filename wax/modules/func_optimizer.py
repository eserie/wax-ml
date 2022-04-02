import haiku as hk
import jax

from wax.modules.transform_params import TransformParams, get_init_params


class FuncOptimizer(hk.Module):
    def __init__(self, func, opt, has_aux=False, name=None):
        self.func = func
        self.opt = opt
        self.has_aux = has_aux
        super().__init__(name=name)

    def __call__(self, *args, **kwargs):
        params = hk.get_state(
            "params", [], init=lambda *_: get_init_params(self.func, *args, **kwargs)
        )
        func_params = TransformParams(self.func)
        l, grads = jax.value_and_grad(func_params, has_aux=self.has_aux)(
            params, *args, **kwargs
        )
        params = jax.tree_multimap(self.opt, params, grads)
        hk.set_state("params", params)
        return l, params
