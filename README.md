# MicroLogState / alfilogd

A deliberately small log-understanding system for monitoring. It combines a frozen sentence-embedding encoder with a tiny fixed-output classifier and a deterministic Nagios renderer.

The model never generates free text and never invents states. It selects from fixed heads:

- `domain`: `FILESYSTEM`, `CPU`, `NETWORK`, `SECURITY`, `MEMORY`, `SERVICE`, `UNKNOWN`
- `health`: `OK` or `BAD`
- `abstain`: classify or return `UNKNOWN`

A local prototype lookup selects a fixed reason such as `high_load`, `oom`, `not_writable`, or `service_down`. The renderer turns the structured result into a Nagios-compatible line.

```text
raw log text
     |
     v
local BGE encoder, frozen
     |
     v
384-dimensional normalized embedding
     |
     v
small shared adapter
     |-- domain head
     |-- health head
     `-- abstain head
     |
     v
reason prototype retrieval
     |
     v
deterministic Nagios renderer
```

Example:

```json
{
  "text": "high load",
  "domain": "CPU",
  "health": "BAD",
  "abstain": false,
  "reason": "high_load",
  "nagios": "CRITICAL - CPU: high CPU load"
}
```

## Repository layout

```text
src/                         v3 classifier, training and inference
data_factory/                local log-template and mutation factory
runtime/alfilogd/            warm systemd runtime daemon
runtime/examples/            main and facility configuration examples
scripts/                     one-shot bootstrap, upgrade and installers
docs/architecture-v3.md      deep technical documentation
docs/runtime-v1.md           runtime and NRPE documentation
```

## Fast path

### 1. Build or upgrade the training environment

```bash
chmod +x scripts/upgrade_micro_log_state_v3.sh
./scripts/upgrade_micro_log_state_v3.sh .
```

This downloads the open BGE encoder once, stores it locally, builds the dataset, trains the tiny classifier and enables offline inference afterwards.

### 2. Build more local training data

```bash
chmod +x scripts/install_log_data_factory_v2.sh
PER_RULE=500 scripts/install_log_data_factory_v2.sh .
```

The data factory uses deterministic templates, Elastic sample formats, existing JSONL corpora, typed parameter pools and controlled permutations. It generates `OK`, `BAD` and `ABSTAIN` examples without using an LLM at runtime.

Additional local corpora can be harvested:

```bash
EXTRA_CORPORA=/var/log:/srv/logarchive \
EXTRA_SEEDS=/data/labeled-windows.jsonl:/data/labeled-linux.jsonl \
scripts/upgrade_micro_log_state_v3.sh .
```

Unlabelled corpora are used only as parameter and grammar sources. They are not blindly treated as ground truth.

### 3. Train

```bash
./run_training_v3.sh
```

### 4. Test inference

```bash
.venv/bin/python -m src.predict --json "filesystem not full writable"
```

### 5. Install the runtime daemon

```bash
sudo scripts/install_alfilogd_runtime_v1.sh \
  --embedding-model ./models/bge-small-en-v1.5 \
  --classifier ./artifacts/micro-log-state-v3.pt \
  --venv ./.venv
```

The runtime keeps the encoder and classifier warm, reads files through inotify, follows journald, supports fallback polling, stores facility state and generates NRPE include files and `get.sh` checks.

## Facility example

```ini
[cpu]
enabled=true
logtype=file
logfile=/var/log/messages
polling=inotify
domain=cpu
output=cpu
recovery_time=60m
```

Nagios/NRPE reads only the generated status wrapper:

```bash
/var/alfilogd/cpu/get.sh
```

It never loads PyTorch.

## What is intentionally not committed

- downloaded Hugging Face models
- trained `.pt` checkpoints
- public or generated datasets
- embedding caches
- virtual environments
- runtime state

These files can be large, have separate licences, or contain local operational data.

## Documentation

- [Architecture and training](docs/architecture-v3.md)
- [Runtime daemon and NRPE](docs/runtime-v1.md)
- [Dataset notes](docs/datasets.md)

## Status

Experimental prototype. The architecture is intentionally constrained, inspectable and designed for local inference rather than general-purpose text generation.
