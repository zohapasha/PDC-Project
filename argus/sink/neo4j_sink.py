"""Optional persistence sink (OFF for all timed experiments).

Consumes hub BelongsTo deltas ("B", tracklet_id, old_person_id, new_person_id) and writes
them to Neo4j in batches with UNWIND, off the critical path:

    (:Tracklet {tid, cam, t_start, t_end})-[:BELONGS_TO]->(:Person {pid})

    pip install neo4j
    sink = Neo4jSink("bolt://localhost:7687", ("neo4j", "password"))
    hub = Hub(topology, engine_config, emit_belongs_to=True)
    ... after handling messages:
        sink.push(hub.engine.emitted_deltas); hub.engine.emitted_deltas.clear()
"""
from __future__ import annotations

from ..common.types import DELTA_BELONGS_TO

UPSERT_QUERY = """
UNWIND $rows AS row
MERGE (t:Tracklet {tid: row.tid})
MERGE (p:Person {pid: row.new})
WITH t, p, row
OPTIONAL MATCH (t)-[old:BELONGS_TO]->(:Person)
DELETE old
MERGE (t)-[:BELONGS_TO]->(p)
"""
CLEANUP_QUERY = "MATCH (p:Person) WHERE NOT (p)<-[:BELONGS_TO]-() DELETE p"


class Neo4jSink:
    def __init__(self, uri, auth, batch_size: int = 5000):
        from neo4j import GraphDatabase
        self.driver = GraphDatabase.driver(uri, auth=auth)
        self.batch_size = batch_size
        self.buffered_rows = []
        with self.driver.session() as session:
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (t:Tracklet) REQUIRE t.tid IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Person) REQUIRE p.pid IS UNIQUE")

    def push(self, deltas):
        for delta in deltas:
            if delta[0] == DELTA_BELONGS_TO:
                _, tracklet_id, _, new_person_id = delta
                self.buffered_rows.append({"tid": tracklet_id, "new": new_person_id})
        if len(self.buffered_rows) >= self.batch_size:
            self.flush()

    def flush(self):
        if not self.buffered_rows:
            return
        # last write per tracklet wins inside a batch
        last_row_of_tracklet = {row["tid"]: row for row in self.buffered_rows}
        with self.driver.session() as session:
            session.run(UPSERT_QUERY, rows=list(last_row_of_tracklet.values()))
            session.run(CLEANUP_QUERY)
        self.buffered_rows = []

    def close(self):
        self.flush()
        self.driver.close()
