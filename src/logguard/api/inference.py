"""Read-only Drain matching and the interface for the model owner's adapter."""
from dataclasses import dataclass
import json
import math
import re
from typing import Protocol

from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence
from drain3.template_miner_config import TemplateMinerConfig

BLOCK = re.compile(r'blk_-?\d+')
HEADER = re.compile(r'^(\d{6})\s+(\d{6})\s+\d+\s+(\w+)\s+([^:]+):\s*(.+)$')


@dataclass(frozen=True)
class Prediction:
    anomaly_score: float
    is_anomaly: bool

    def __post_init__(self):
        if not math.isfinite(self.anomaly_score):
            raise ValueError('Prediction score must be finite')


class Predictor(Protocol):
    # Encoded indices in chronological order, including the current event.
    # At most window_size + 1 entries; PAD and model-specific shaping belong to the adapter.
    def predict(self, events: tuple[int, ...]) -> Prediction: ...


class LogMatcher:
    def __init__(self, settings):
        for key in ('drain_config', 'drain_state', 'vocab'):
            if not settings[key].is_file():
                raise FileNotFoundError(f'Missing {key}: {settings[key]}')
        config = TemplateMinerConfig()
        config.load(str(settings['drain_config']))
        self.miner = TemplateMiner(FilePersistence(str(settings['drain_state'])), config)
        self.vocab = json.loads(settings['vocab'].read_text())['event_to_index']

    def parse(self, raw):
        match = HEADER.fullmatch(raw.strip())
        if not match:
            raise ValueError('Expected one HDFS log line: YYMMDD HHMMSS PID LEVEL component: message')
        date, time, level, component, content = match.groups()
        cluster = self.miner.match(content)
        event_id = cluster.cluster_id if cluster else None
        return {
            'log_date': date, 'log_time': time, 'level': level, 'component': component,
            'event_id': event_id, 'event_index': self.vocab.get(str(event_id), 1),
        }, list(dict.fromkeys(BLOCK.findall(content)))
