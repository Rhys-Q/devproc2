"""EmitExecutablePass — serialize VM Executable to executable.vm artifact file."""
from __future__ import annotations

import os

from devproc2.vm import serializer
from devproc2.vm.executable import Executable


class EmitExecutablePass:
    """Serialize an Executable to <output_dir>/executable.vm.

    Returns the raw bytes so callers can verify round-trips without disk I/O.
    """

    def run(self, exe: Executable, output_dir: str) -> bytes:
        data = serializer.serialize(exe)
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "executable.vm")
        with open(path, "wb") as f:
            f.write(data)
        return data
