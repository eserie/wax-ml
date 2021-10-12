import haiku as hk
import jax


class VMap(hk.Module):
    def __init__(self, fun, name=None):
        super().__init__(name=name)
        self.fun = (
            fun
            if isinstance(fun, hk.TransformedWithState)
            else hk.transform_with_state(fun)
        )

    def __call__(self, *args, **kwargs):
        n_batches = len(jax.tree_leaves((args, kwargs))[0])
        rng = hk.next_rng_key()
        rng = jax.random.split(rng, num=n_batches)
        params, state = hk.get_state(
            "params_state", [], init=lambda *_: jax.vmap(self.fun.init)(rng, *args, **kwargs)
        )
        res, state = jax.vmap(self.fun.apply)(params, state, rng, *args, **kwargs)
        hk.set_state("params_state", (params, state))

        return res
    