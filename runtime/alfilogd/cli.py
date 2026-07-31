from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import load_config


def parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


async def socket_request(socket_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    reader, writer = await asyncio.open_unix_connection(str(socket_path))
    writer.write(json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n")
    await writer.drain()
    raw = await reader.readline()
    writer.close()
    await writer.wait_closed()
    if not raw:
        raise RuntimeError("alfilogd closed the socket without a response")
    response = json.loads(raw)
    if not response.get("ok"):
        raise RuntimeError(str(response.get("error", "unknown daemon error")))
    return response


def emit_result(result: dict[str, Any], json_output: bool, nagios: bool = False) -> int:
    if json_output:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif nagios:
        print(result.get("nagios", "UNKNOWN - malformed alfilogd response"))
    else:
        print(result.get("nagios", json.dumps(result, ensure_ascii=False)))
    return int(result.get("nagios_code", 3)) if nagios else 0


def command_status(args: argparse.Namespace) -> int:
    if args.status_file:
        try:
            payload = json.loads(args.status_file.read_text(encoding="utf-8"))
        except FileNotFoundError:
            print(f"UNKNOWN - status file missing: {args.status_file}")
            return 3
        except (OSError, json.JSONDecodeError) as exc:
            print(f"UNKNOWN - cannot read status: {exc}")
            return 3
        updated = parse_iso(payload.get("daemon_updated_at"))
        if args.max_age > 0 and updated is not None:
            age = (datetime.now(timezone.utc) - updated.astimezone(timezone.utc)).total_seconds()
            if age > args.max_age:
                print(f"UNKNOWN - status stale, last daemon update {int(age)} seconds ago")
                return 3
        return emit_result(payload, args.json, args.nagios)

    if not args.facility:
        raise SystemExit("status requires --status-file or --facility")
    response = asyncio.run(
        socket_request(args.socket, {"action": "status", "facility": args.facility})
    )
    return emit_result(response["result"], args.json, args.nagios)


def command_classify(args: argparse.Namespace) -> int:
    texts: list[str] = []
    if args.stdin:
        texts.extend(line.rstrip("\n") for line in sys.stdin if line.strip())
    if args.text:
        texts.append(" ".join(args.text))
    if not texts:
        raise SystemExit("provide text or --stdin")

    async def classify_chunks() -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for offset in range(0, len(texts), max(1, args.batch_size)):
            chunk = texts[offset : offset + max(1, args.batch_size)]
            if len(chunk) == 1:
                response = await socket_request(
                    args.socket,
                    {"action": "classify", "text": chunk[0], "source": args.source},
                )
                output.append(response["result"])
            else:
                response = await socket_request(
                    args.socket,
                    {"action": "classify_batch", "texts": chunk, "source": args.source},
                )
                output.extend(response["results"])
        return output

    results = asyncio.run(classify_chunks())
    return_code = 0
    for result in results:
        code = emit_result(result, args.json, args.nagios)
        if args.nagios:
            return_code = max(return_code, code)
    return return_code


def command_ping(args: argparse.Namespace) -> int:
    response = asyncio.run(socket_request(args.socket, {"action": "ping"}))
    print(json.dumps(response, ensure_ascii=False, indent=2) if args.json else "PONG")
    return 0


def command_validate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    payload = {
        "embedding_model": str(config.main.embedding_model),
        "classifier": str(config.main.classifier),
        "facilities": {
            name: {
                "logtype": facility.logtype,
                "polling": facility.polling,
                "domain": facility.domain,
                "output": facility.output,
                "source_key": list(facility.source_key),
            }
            for name, facility in config.facilities.items()
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="alfilogctl")
    parser.set_defaults(func=None)
    sub = parser.add_subparsers(dest="command")

    classify = sub.add_parser("classify", help="classify text through the warm daemon")
    classify.add_argument("text", nargs="*")
    classify.add_argument("--stdin", action="store_true")
    classify.add_argument("--source", default="alfilogctl")
    classify.add_argument("--batch-size", type=int, default=64)
    classify.add_argument("--socket", type=Path, default=Path("/run/alfilogd/alfilogd.sock"))
    classify.add_argument("--json", action="store_true")
    classify.add_argument("--nagios", action="store_true")
    classify.set_defaults(func=command_classify)

    status = sub.add_parser("status", help="read facility status")
    status.add_argument("--facility")
    status.add_argument("--status-file", type=Path)
    status.add_argument("--socket", type=Path, default=Path("/run/alfilogd/alfilogd.sock"))
    status.add_argument("--max-age", type=float, default=0)
    status.add_argument("--json", action="store_true")
    status.add_argument("--nagios", action="store_true")
    status.set_defaults(func=command_status)

    ping = sub.add_parser("ping")
    ping.add_argument("--socket", type=Path, default=Path("/run/alfilogd/alfilogd.sock"))
    ping.add_argument("--json", action="store_true")
    ping.set_defaults(func=command_ping)

    validate = sub.add_parser("validate-config")
    validate.add_argument("--config", type=Path, default=Path("/etc/alfilogd/alfilogd.conf"))
    validate.set_defaults(func=command_validate)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.func is None:
        parser.print_help()
        return 2
    try:
        return int(args.func(args))
    except (OSError, RuntimeError) as exc:
        print(f"UNKNOWN - {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
