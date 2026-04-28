"""
Neo4j Schema — Cypher queries for index creation and schema management.

Vector index dimensions are no longer hardcoded; they come from
`Config.EMBEDDING_DIMENSIONS`. Use `build_schema_queries(dimensions)` to
materialise the schema for a specific embedding model.

Called by Neo4jStorage._ensure_schema().
"""

from typing import List


# Constraints (dimension-independent)
CREATE_GRAPH_UUID_CONSTRAINT = """
CREATE CONSTRAINT graph_uuid IF NOT EXISTS
FOR (g:Graph) REQUIRE g.graph_id IS UNIQUE
"""

CREATE_ENTITY_UUID_CONSTRAINT = """
CREATE CONSTRAINT entity_uuid IF NOT EXISTS
FOR (n:Entity) REQUIRE n.uuid IS UNIQUE
"""

CREATE_EPISODE_UUID_CONSTRAINT = """
CREATE CONSTRAINT episode_uuid IF NOT EXISTS
FOR (ep:Episode) REQUIRE ep.uuid IS UNIQUE
"""

# Fulltext indexes (for BM25 keyword search; dimension-independent)
CREATE_ENTITY_FULLTEXT_INDEX = """
CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS
FOR (n:Entity) ON EACH [n.name, n.summary]
"""

CREATE_FACT_FULLTEXT_INDEX = """
CREATE FULLTEXT INDEX fact_fulltext IF NOT EXISTS
FOR ()-[r:RELATION]-() ON EACH [r.fact, r.name]
"""

# Index names — exposed so callers can inspect dimension via SHOW INDEXES.
ENTITY_VECTOR_INDEX_NAME = "entity_embedding"
RELATION_VECTOR_INDEX_NAME = "fact_embedding"


def _vector_index_query(name: str, target: str, prop: str, dimensions: int) -> str:
    return f"""
    CREATE VECTOR INDEX {name} IF NOT EXISTS
    FOR {target} ON ({prop})
    OPTIONS {{indexConfig: {{
        `vector.dimensions`: {int(dimensions)},
        `vector.similarity_function`: 'cosine'
    }}}}
    """


def build_schema_queries(dimensions: int) -> List[str]:
    """Return all schema queries parameterised by `dimensions`."""
    if dimensions <= 0:
        raise ValueError(f"dimensions must be > 0 (got {dimensions})")

    return [
        CREATE_GRAPH_UUID_CONSTRAINT,
        CREATE_ENTITY_UUID_CONSTRAINT,
        CREATE_EPISODE_UUID_CONSTRAINT,
        _vector_index_query(
            ENTITY_VECTOR_INDEX_NAME,
            "(n:Entity)",
            "n.embedding",
            dimensions,
        ),
        _vector_index_query(
            RELATION_VECTOR_INDEX_NAME,
            "()-[r:RELATION]-()",
            "r.fact_embedding",
            dimensions,
        ),
        CREATE_ENTITY_FULLTEXT_INDEX,
        CREATE_FACT_FULLTEXT_INDEX,
    ]
