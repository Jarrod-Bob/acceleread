# SPDX-License-Identifier: Apache-2.0
import pytest

import acceleread
from acceleread.cli import main


def test_version_is_set() -> None:
    assert acceleread.__version__


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert acceleread.__version__ in capsys.readouterr().out
