"""Install a project-local GNU Make 4.3 on Intel macOS without Xcode."""

import hashlib
import io
import platform
import subprocess
import tarfile
import urllib.request
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    target = root / ".venv" / "bin" / "make"
    if target.exists():
        subprocess.run([str(target), "--version"], check=True)
        return
    if platform.system() != "Darwin" or platform.machine() != "x86_64":
        raise SystemExit("Use GNU Make >=4.3 from your package manager on this platform.")
    url = "https://conda.anaconda.org/conda-forge/osx-64/make-4.3-h22f3db7_1.tar.bz2"
    with urllib.request.urlopen(url, timeout=60) as response:
        archive_bytes = response.read()
    # Published conda-forge archive checksum; HTTPS is used for acquisition.
    if hashlib.md5(archive_bytes).hexdigest() != "ac4a1dd58e6d821c518ae0011e8592b7":
        raise ValueError("GNU Make archive checksum mismatch")
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:bz2") as archive:
        member = archive.getmember("bin/make")
        if not member.isfile():
            raise ValueError("Expected a regular make executable")
        binary = archive.extractfile(member).read()
    temporary = target.with_name("make.part")
    temporary.write_bytes(binary)
    temporary.chmod(0o755)
    subprocess.run([str(temporary), "--version"], check=True)
    temporary.replace(target)
    print(f"Installed {target} from {url}")


if __name__ == "__main__":
    main()
