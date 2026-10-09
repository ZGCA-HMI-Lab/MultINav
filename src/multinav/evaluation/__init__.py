"""Standalone evaluator for the ``interactive_nav_v3`` benchmark.

This package deliberately lives outside ``molmo_spaces.evaluation`` so the
evaluator contract can evolve independently of the simulator. A simulator is
reached only through :mod:`multinav.platforms`; ROS-specific helpers are kept
behind optional adapters.
"""
