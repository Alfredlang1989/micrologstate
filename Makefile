PYTHON ?= .venv/bin/python
DATASET ?= data/processed/log_states_v3.jsonl
ENCODER ?= models/bge-small-en-v1.5

.PHONY: train demo inspect compile

train:
	$(PYTHON) -m src.train --data "$(DATASET)" --encoder-path "$(ENCODER)"

demo:
	$(PYTHON) -m src.predict --json "high load"
	$(PYTHON) -m src.predict --json "filesystem not full writable"
	$(PYTHON) -m src.predict --json "nginx.service entered failed state"

inspect:
	$(PYTHON) -m src.inspect_vectors "filesystem capacity exhausted"

compile:
	$(PYTHON) -m compileall -q src data_factory runtime/alfilogd
