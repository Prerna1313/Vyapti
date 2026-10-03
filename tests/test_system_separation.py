"""System implementations live at one canonical path and start independently."""
import importlib
import subprocess
import sys

import pytest


@pytest.mark.parametrize("module_name", [
    "vyapti_simulator.system_b.tsrd.tsrd_environment",
    "vyapti_simulator.system_b.tsrd.train250_cache",
    "vyapti_simulator.system_b.tsrd.policies.classical_ucb1",
    "vyapti_simulator.system_c.rf.pulse_detector",
    "vyapti_simulator.system_c.rf.simulator_engine",
    "vyapti_simulator.system_c.emitters.emitter_models",
    "vyapti_simulator.system_c.emitters.rf_pulse_simulator",
    "vyapti_simulator.core.observation_interface",
])
def test_canonical_modules_are_inside_simulator_package(module_name):
    module = importlib.import_module(module_name)
    assert "vyapti_simulator" in module.__file__.replace("\\", "/")


@pytest.mark.parametrize("imports, forbidden", [
    (["vyapti_simulator.system_b.tsrd.train250_cache", "vyapti_simulator.system_b.tsrd.experiment"],
     ["vyapti_simulator.system_c", "src"]),
    (["vyapti_simulator.system_c.rf.rf_to_pdw_pipeline", "vyapti_simulator.system_c.emitters.rf_pulse_simulator"],
     ["vyapti_simulator.system_b"]),
])
def test_system_starts_without_loading_other_system(imports, forbidden):
    code = (
        "import importlib, sys\n"
        f"for name in {imports!r}: importlib.import_module(name)\n"
        f"bad = [name for name in sys.modules if any(name == p or name.startswith(p + '.') for p in {forbidden!r})]\n"
        "assert not bad, bad\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
