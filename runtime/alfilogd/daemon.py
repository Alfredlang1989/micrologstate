from __future__ import annotations

import argparse
import asyncio
import grp
import json
import logging
import os
import signal
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import FacilityConfig, RuntimeConfig, load_config
from .model_engine import ModelEngine
from .nrpe import generate_runtime_outputs
from .ontology import Classification
from .sources import CursorStore, FileInotifySource, FilePollSource, JournalSource
from .state import LogEvent, StateManager

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class QueueItem:
    event: LogEvent
    targets: tuple[str, ...]
    response: asyncio.Future[Classification] | None = None


def classification_payload(result: Classification) -> dict[str, Any]:
    return {
        "text": result.text,
        "domain": result.domain,
        "health": None if result.health is None else ("OK" if result.health == 0 else "BAD"),
        "health_value": result.health,
        "abstain": result.abstain,
        "reason": result.reason,
        "reason_confidence": result.reason_confidence,
        "domain_confidence": result.domain_confidence,
        "health_confidence": result.health_confidence,
        "abstain_probability": result.abstain_probability,
        "accept_score": result.accept_score,
        "nagios_code": result.nagios_code,
        "nagios": result.nagios,
    }


class AlfilogDaemon:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.stop_event = asyncio.Event()
        self.queue: asyncio.Queue[QueueItem] = asyncio.Queue(maxsize=config.main.queue_size)
        self.state = StateManager(config.main, config.facilities)
        self.cursor_store = CursorStore(config.main.state_base / "cursors")
        self.model = ModelEngine(
            config.main.embedding_model,
            config.main.classifier,
            config.main.device,
        )
        self.tasks: list[asyncio.Task[Any]] = []
        self.socket_server: asyncio.AbstractServer | None = None
        self.dropped_events = 0

    async def enqueue(self, event: LogEvent, targets: tuple[str, ...]) -> None:
        item = QueueItem(event=event, targets=targets)
        try:
            self.queue.put_nowait(item)
        except asyncio.QueueFull:
            self.dropped_events += 1
            if self.dropped_events == 1 or self.dropped_events % 100 == 0:
                LOGGER.error("classification queue full; dropped events=%s", self.dropped_events)

    def build_sources(self) -> list[Any]:
        grouped: dict[tuple[object, ...], list[FacilityConfig]] = {}
        for facility in self.config.facilities.values():
            if facility.enabled:
                grouped.setdefault(facility.source_key, []).append(facility)
        sources: list[Any] = []
        for facilities in grouped.values():
            template = facilities[0]
            targets = tuple(item.name for item in facilities)
            if template.logtype == "file" and template.polling == "inotify":
                source = FileInotifySource(
                    self.config.main, template, targets, self.enqueue, self.cursor_store
                )
            elif template.logtype == "file" and template.polling == "poll":
                source = FilePollSource(
                    self.config.main, template, targets, self.enqueue, self.cursor_store
                )
            elif template.logtype == "systemd":
                source = JournalSource(
                    self.config.main, template, targets, self.enqueue, self.cursor_store
                )
            else:
                raise ValueError(f"unsupported source: {template}")
            sources.append(source)
            LOGGER.info("source %s targets=%s", template.source_key, ",".join(targets))
        return sources

    def matches(self, facility_name: str, result: Classification) -> bool:
        facility = self.config.facilities[facility_name]
        return facility.domain == "ANY" or facility.domain == result.domain

    async def classifier_loop(self) -> None:
        batch_size = self.config.main.batch_size
        wait_seconds = self.config.main.batch_wait_ms / 1000.0
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                first = await asyncio.wait_for(self.queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            batch = [first]
            deadline = asyncio.get_running_loop().time() + wait_seconds
            while len(batch) < batch_size:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self.queue.get(), timeout=remaining))
                except asyncio.TimeoutError:
                    break
            texts = [item.event.text for item in batch]
            try:
                results = await asyncio.to_thread(self.model.classify_batch, texts)
                for item, result in zip(batch, results, strict=True):
                    for facility_name in item.targets:
                        if self.matches(facility_name, result):
                            self.state.apply(facility_name, item.event, result)
                    if item.response is not None and not item.response.done():
                        item.response.set_result(result)
            except Exception as exc:
                LOGGER.exception("inference batch failed")
                for item in batch:
                    if item.response is not None and not item.response.done():
                        item.response.set_exception(exc)
            finally:
                for _ in batch:
                    self.queue.task_done()

    async def status_loop(self) -> None:
        while not self.stop_event.is_set():
            self.state.expire_all()
            self.state.write_all()
            try:
                await asyncio.wait_for(
                    self.stop_event.wait(), timeout=self.config.main.status_write_interval
                )
            except asyncio.TimeoutError:
                pass
        self.state.expire_all()
        self.state.write_all()

    async def classify_requests(self, texts: list[str], source: str = "socket") -> list[Classification]:
        futures: list[asyncio.Future[Classification]] = []
        now = datetime.now(timezone.utc)
        for text in texts:
            future: asyncio.Future[Classification] = asyncio.get_running_loop().create_future()
            futures.append(future)
            event = LogEvent(
                text=text[: self.config.main.max_line_bytes],
                source=source,
                observed_at=now,
                metadata={"logtype": "socket"},
            )
            await self.queue.put(QueueItem(event=event, targets=(), response=future))
        return list(await asyncio.gather(*futures))

    async def classify_request(self, text: str, source: str = "socket") -> Classification:
        return (await self.classify_requests([text], source))[0]

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while not reader.at_eof():
                raw = await reader.readline()
                if not raw:
                    break
                try:
                    request = json.loads(raw)
                    action = str(request.get("action", "classify"))
                    if action == "ping":
                        response: dict[str, Any] = {
                            "ok": True,
                            "version": 1,
                            "queue_size": self.queue.qsize(),
                            "dropped_events": self.dropped_events,
                            "device": self.model.device_name,
                        }
                    elif action == "classify":
                        text = str(request.get("text", ""))
                        if not text.strip():
                            raise ValueError("classify requires non-empty text")
                        result = await self.classify_request(text, str(request.get("source", "socket")))
                        response = {"ok": True, "result": classification_payload(result)}
                    elif action == "classify_batch":
                        texts = [str(item) for item in request.get("texts", []) if str(item).strip()]
                        if not texts:
                            raise ValueError("classify_batch requires non-empty texts")
                        if len(texts) > 1024:
                            raise ValueError("classify_batch maximum is 1024 texts")
                        results = await self.classify_requests(texts, str(request.get("source", "socket")))
                        response = {
                            "ok": True,
                            "results": [classification_payload(item) for item in results],
                        }
                    elif action == "status":
                        facility = str(request.get("facility", ""))
                        snapshot = self.state.get_snapshot(facility)
                        if snapshot is None:
                            raise KeyError(f"unknown facility: {facility}")
                        response = {"ok": True, "result": snapshot}
                    else:
                        raise ValueError(f"unknown action: {action}")
                except Exception as exc:
                    response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                writer.write(json.dumps(response, ensure_ascii=False).encode("utf-8") + b"\n")
                await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    async def start_socket(self) -> None:
        socket_path = self.config.main.socket_path
        socket_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            socket_path.unlink()
        except FileNotFoundError:
            pass
        self.socket_server = await asyncio.start_unix_server(
            self.handle_client, path=str(socket_path)
        )
        os.chmod(socket_path, self.config.main.socket_mode)
        try:
            group_id = grp.getgrnam(self.config.main.socket_group).gr_gid
            os.chown(socket_path, -1, group_id)
        except KeyError:
            LOGGER.warning("socket group does not exist: %s", self.config.main.socket_group)
        LOGGER.info("socket listening on %s", socket_path)

    async def run(self) -> None:
        generate_runtime_outputs(self.config)
        self.state.write_all()
        await self.start_socket()
        self.tasks.append(asyncio.create_task(self.classifier_loop(), name="classifier"))
        self.tasks.append(asyncio.create_task(self.status_loop(), name="state-writer"))
        for index, source in enumerate(self.build_sources(), start=1):
            self.tasks.append(
                asyncio.create_task(source.run(self.stop_event), name=f"source-{index}")
            )
        LOGGER.info(
            "alfilogd started: facilities=%s batch=%s/%sms device=%s",
            len(self.state.states),
            self.config.main.batch_size,
            self.config.main.batch_wait_ms,
            self.model.device_name,
        )
        await self.stop_event.wait()
        LOGGER.info("shutdown requested")
        for task in self.tasks:
            if task.get_name().startswith("source-"):
                task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        if self.socket_server is not None:
            self.socket_server.close()
            await self.socket_server.wait_closed()
        try:
            self.config.main.socket_path.unlink()
        except FileNotFoundError:
            pass

    def stop(self) -> None:
        self.stop_event.set()


async def async_main(config_path: Path) -> int:
    config = load_config(config_path)
    daemon = AlfilogDaemon(config)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            loop.add_signal_handler(sig, daemon.stop)
        except NotImplementedError:
            pass
    await daemon.run()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Alfilog model daemon")
    parser.add_argument("--config", type=Path, default=Path("/etc/alfilogd/alfilogd.conf"))
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return asyncio.run(async_main(args.config))


if __name__ == "__main__":
    raise SystemExit(main())
