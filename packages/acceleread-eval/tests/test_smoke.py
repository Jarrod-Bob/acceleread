# SPDX-License-Identifier: Apache-2.0
from acceleread_eval.cli import main


def test_cli_runs() -> None:
    assert main([]) == 0
