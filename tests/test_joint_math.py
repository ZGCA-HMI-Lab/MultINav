"""The articulation convention shared by generation, execution and scoring.

Generation, force execution and scoring must agree on which joint endpoint is
"closed", otherwise an episode that genuinely opened a door can be scored as
closed. These tests pin that convention, plus the MuJoCo-bound lookups.
"""

from __future__ import annotations

import pytest

from multinav.core.joints import (
    joint_closed_open_values,
    joint_range_by_name,
    joint_value_by_name,
    semantic_open_fraction,
)

SIMPLE_MODEL_XML = """
<mujoco>
  <worldbody>
    <body name="body">
      <joint name="slider" type="slide" range="-1.2 0.3"/>
      <geom type="box" size="0.1 0.1 0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_closed_endpoint_is_the_value_nearest_zero():
    assert joint_closed_open_values([0.0, 1.5]) == (0.0, 1.5)
    assert joint_closed_open_values([-1.2, 0.3]) == (0.3, -1.2)
    assert joint_closed_open_values([0.4, 1.1]) == (0.4, 1.1)


def test_open_fraction_spans_the_closed_open_endpoints():
    assert semantic_open_fraction(0.0, 0.0, 1.5) == 0.0
    assert semantic_open_fraction(1.5, 0.0, 1.5) == 1.0
    assert semantic_open_fraction(0.75, 0.0, 1.5) == pytest.approx(0.5)


def test_open_fraction_is_clamped_and_survives_a_zero_span():
    assert semantic_open_fraction(2.0, 0.0, 1.5) == 1.0
    assert semantic_open_fraction(-1.0, 0.0, 1.5) == 0.0
    assert semantic_open_fraction(0.5, 0.5, 0.5) == 0.0


def test_named_lookups_read_the_configured_joint():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(SIMPLE_MODEL_XML)
    data = mujoco.MjData(model)

    assert joint_range_by_name(model, "slider") == (-1.2, 0.3)
    assert joint_value_by_name(model, data, "slider") == 0.0


def test_named_lookups_reject_unknown_joints():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(SIMPLE_MODEL_XML)

    with pytest.raises(ValueError, match="Joint not found: missing"):
        joint_range_by_name(model, "missing")
