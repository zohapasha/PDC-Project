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
    __slots__ = ("queue", "n_messages", "touched_vertices", "handlers")

    def __init__(self):
        self.queue = deque()
        self.n_messages = 0
        self.touched_vertices = set()
        self.handlers = {}

    def register(self, message_kind: int, handler):
        self.handlers[message_kind] = handler

    def reset(self):
        self.n_messages = 0
        self.touched_vertices = set()

    def send(self, vertex_id: int, message_kind: int, payload):
        self.queue.append((vertex_id, message_kind, payload))
        self.n_messages += 1

    def run(self):
        queue, handlers, touched_vertices = self.queue, self.handlers, self.touched_vertices
        while queue:
            vertex_id, message_kind, payload = queue.popleft()
            touched_vertices.add(vertex_id)
            handlers[message_kind](vertex_id, payload)

    def record_local_work(self, vertex_id: int):
        """Account for local work at a vertex that needs no message (e.g. the root
        updating its person summary)."""
        self.touched_vertices.add(vertex_id)


# message kinds
CLIMB, CLIMB_STOP, MARK, RELABEL, REROOT, CHECK_BACKUP = range(6)
