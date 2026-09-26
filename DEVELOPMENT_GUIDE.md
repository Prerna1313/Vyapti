# Vyapti Simulator Development Guide

Check [REPOSITORY_STATUS.md](REPOSITORY_STATUS.md) before relying on older
implementation notes or test counts.

## Setting up the Development Environment

1. Clone the repository
2. Install the declared development dependencies: `pip install -e ".[dev]"`

## Running Tests

- Run all tests: `python -m pytest -q`
- Run a specific module: `python -m pytest tests/test_rf_pulse_detector.py -q`
- Check maintained documentation: `python -m pytest tests/test_documentation_contract.py -q`
- Check syntax: `python -m compileall -q vyapti_simulator src tests`
- Check patch whitespace: `git diff --check`

## Code Style

Keep changes readable and avoid unrelated formatting in an existing dirty tree.
No repository-wide formatter or pre-commit configuration is currently declared.

## Making Changes

1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Documentation

Update docstrings for any new or changed functions.
For major changes, update the relevant .md files in the root directory.
