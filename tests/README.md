# tests

`test_run_all_flow.py` checks `run_all.py` without the competition data: every heavy function of `tfb/` is replaced by
a stub that validates its call against the real function's signature and writes the files the real one writes. It
checks the stage order, the arguments, the chosen component files and that a second run reuses every finished stage.

    python tests/test_run_all_flow.py
