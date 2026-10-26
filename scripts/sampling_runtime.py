"""Small atomic sampling status files."""
import json
import os
from pathlib import Path
import time



class SamplingProgress:
    def __init__(self, path, dataset, split):
        self.path = Path(path) if path else None
        self.dataset, self.split = dataset, split
        self.started_at = None
        self.total = None
        self.completed = 0
        self.positive = 0

    def update(self, phase):
        if self.path is None:
            return
        now = time.time()
        elapsed = now - self.started_at if self.started_at else 0
        rate = self.completed / elapsed if elapsed > 0 else 0
        remaining = (self.total - self.completed) / rate if rate and self.total is not None else None
        state = dict(dataset=self.dataset, split=self.split, phase=phase,
                     pid=os.getpid(), updated_at=now, started_at=self.started_at,
                     total=self.total, completed=self.completed, positive=self.positive,
                     questions_per_second=rate, remaining_seconds=remaining)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
        os.replace(temporary, self.path)
