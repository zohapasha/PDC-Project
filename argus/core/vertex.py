"""Per-process vertex-program scheduler.

Every traversal in the engine (path climbing, subtree marking, relabelling, rerooting,
backup checks) is written as message handlers on vertices. A handler only reads and
writes the state of the vertex it runs on and sends messages to tree neighbours.
The scheduler counts messages and the set of touched vertices per operation -- these
are the numbers that back the claim "a correction touches only one person".
"""
from __future__ import annotations

from collections import deque


class VertexScheduler:
    __slots__ = ("q", "msgs", "touched", "handlers")

    def __init__(self):
        self.q = deque()
        self.msgs = 0
        self.touched = set()
        self.handlers = {}

    def register(self, kind: int, fn):
        self.handlers[kind] = fn

    def reset(self):
        self.msgs = 0
        self.touched = set()

    def send(self, vid: int, kind: int, payload):
        self.q.append((vid, kind, payload))
        self.msgs += 1

    def run(self):
        q, handlers, touched = self.q, self.handlers, self.touched
        while q:
            vid, kind, payload = q.popleft()
            touched.add(vid)
            handlers[kind](vid, payload)

    def local(self, vid: int):
        """Account for local work at a vertex that needs no message (e.g. the root
        updating its person summary)."""
        self.touched.add(vid)


# message kinds
CLIMB, CLIMB_STOP, MARK, RELABEL, REROOT, CHECK_BACKUP = range(6)
