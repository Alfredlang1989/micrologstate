# Dataset notes

The starter run uses balanced samples rather than downloading every row.
`datasets.load_dataset(..., streaming=True)` progressively reads the public
Parquet data and writes the selected rows into `data/processed/log_states.jsonl`.

- `logfit-project/HDFS_v1`: HDFS console logs, binary anomaly label.
- `logfit-project/BGL`: BlueGene/L console logs, alert/anomaly label.
- `witfoo/precinct6-cybersecurity`, config `signals`: sanitized enterprise
  security events with benign/suspicious/malicious labels.

Important: BGL and HDFS are marked `license: other` on their Hugging Face cards.
Review the original dataset terms before redistributing the downloaded rows.
WitFoo is marked Apache-2.0. Its labels are produced mainly by an automated
correlation engine and are therefore treated here as `weak-oracle` labels,
not unquestionable ground truth.
