"""Add two industrial-route guards to the inspected, already deployed PC service."""

import argparse
import hashlib
from pathlib import Path


EXPECTED_SHA256 = "5329783224abfaf3067d800811baaf74f6b4a747b242c5cd23117ebcf6b864ee"
IMPORT = "from frontend_static import handle_frontend_get\n"
GET = "    def do_GET(self) -> None:                            # noqa: N802\n"
POST = "    def do_POST(self) -> None:                           # noqa: N802\n"


def patched_source(original):
    if hashlib.sha256(original).hexdigest() != EXPECTED_SHA256:
        raise ValueError("Live Sentinel source changed; inspect it before generating a new patch")
    source = original.decode("utf-8")
    for anchor in (IMPORT, GET, POST):
        if source.count(anchor) != 1:
            raise ValueError("Expected unique import/Handler route anchors")
    source = source.replace(IMPORT, IMPORT + "from industrial_runtime import handle_industrial_get, handle_industrial_post\n", 1)
    source = source.replace(GET, GET + "        if handle_industrial_get(self, ROOT):\n            return\n", 1)
    source = source.replace(POST, POST + "        if handle_industrial_post(self, ROOT):\n            return\n", 1)
    compile(source, "sentinel-with-industrial.py", "exec")
    return source.encode("utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        parser.error("Use a new output path; the source is never edited")
    try:
        data = patched_source(args.source.read_bytes())
        with args.output.open("xb") as output:
            output.write(data)
    except (OSError, ValueError) as error:
        parser.exit(1, str(error) + "\n")
    print(hashlib.sha256(data).hexdigest(), args.output)
