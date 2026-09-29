from __future__ import annotations

import argparse
import json
from pathlib import Path

from .engine import DecisionEngine
from .models import DecisionRequest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an offline JEV-style decision")
    parser.add_argument("--conversation-id", default="cli-demo")
    parser.add_argument("--message", required=True)
    args = parser.parse_args()
    result = DecisionEngine().decide(
        DecisionRequest(conversation_id=args.conversation_id, current_message=args.message)
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
