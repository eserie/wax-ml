
wax.modules package
===================

.. currentmodule:: wax.modules

.. automodule:: wax.modules

Gym modules
-------------

In WAX-ML, an agent and environments are simple functions:

.. figure:: tikz/agent_env.png
    :scale: 25 %
    :align: center


A Gym feedback loops can be represented with the diagram:

.. figure:: tikz/gymfeedback.png
    :scale: 25 %
    :align: center

Equivalently, it can be described with the pair of `init` and `apply` functions:

.. figure:: tikz/gymfeedback_init_apply.png
    :scale: 25 %
    :align: center


.. autosummary::
  :toctree: _autosummary

	gym_feedback


Online Learning
-------------------

WAX-ML contains a module to perform online learning for supervised problems.

.. autosummary::
  :toctree: _autosummary

   online_supervised_learner


Other Haiku modules
------------------------

.. autosummary::
  :toctree: _autosummary

	buffer
	diff
	ewma
	ewmcov
	ewmvar
	has_changed
	lag
	ohlc
	pct_change
	rolling_mean
	update_on_event
