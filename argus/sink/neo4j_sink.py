"""Optional persistence sink (OFF for all timed experiments).

Consumes hub BelongsTo deltas ("B", tid, old_pid, new_pid) and writes them to Neo4j in
batches with UNWIND, off the critical path:

    (:Tracklet {tid, cam, t_start, t_end})-[:BELONGS_TO]->(:Person {pid})

    pip install neo4j
    sink = Neo4jSink("bolt://localhost:7687", ("neo4j", "password"))
    hub = Hub(topo, cfg, emit_belongs=True)
    ... after handling messages:  sink.push(hub.eng.out); hub.eng.out.clear()
"""
from __future__ import annotations

UPSERT = """
UNWIND $rows AS r
MERGE (t:Tracklet {tid: r.tid})
MERGE (p:Person {pid: r.new})
WITH t, p, r
OPTIONAL MATCH (t)-[old:BELONGS_TO]->(:Person)
DELETE old
MERGE (t)-[:BELONGS_TO]->(p)
"""
CLEANUP = "MATCH (p:Person) WHERE NOT (p)<-[:BELONGS_TO]-() DELETE p"


class Neo4jSink:
    def __init__(self, uri, auth, batch: int = 5000):
        from neo4j import GraphDatabase
        self.driver = GraphDatabase.driver(uri, auth=auth)
        self.batch = batch
        self.buf = []
        with self.driver.session() as s:
            s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (t:Tracklet) REQUIRE t.tid IS UNIQUE")
            s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Person) REQUIRE p.pid IS UNIQUE")

    def push(self, deltas):
        for d in deltas:
            if d[0] == "B":
                self.buf.append({"tid": d[1], "new": d[3]})
        if len(self.buf) >= self.batch:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        # last write per tracklet wins inside a batch
        last = {r["tid"]: r for r in self.buf}
        with self.driver.session() as s:
            s.run(UPSERT, rows=list(last.values()))
            s.run(CLEANUP)
        self.buf = []

    def close(self):
        self.flush()
        self.driver.close()
