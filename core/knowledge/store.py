"""LanceDB-backed knowledge store for schema, queries, and business context."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass
class KnowledgeDoc:
    id: str
    text: str
    metadata: dict
    category: str  # "schema" | "query" | "metric" | "business_rule"


class KnowledgeStore:
    """Vector store for data engineering knowledge: schema descriptions, reference SQL, business metrics, business rules."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._table = None
        self._embed, self._embed_many = None, None

    def _lazy_init(self):
        if self._table is not None:
            return
        import lancedb
        db = lancedb.connect(str(self.db_path))
        try:
            self._table = db.open_table("knowledge")
        except Exception:
            self._table = db.create_table("knowledge", [
                {"vector": [0.0] * 384, "id": "", "text": "", "metadata": "{}", "category": ""}
            ], mode="overwrite")
        from core.knowledge.embeddings import get_embedder
        self._embed, self._embed_many = get_embedder()

    def add_doc(self, doc: KnowledgeDoc):
        """Add a single document to the knowledge base."""
        self._lazy_init()
        vec = self._embed(doc.text)
        self._table.add([{
            "vector": vec, "id": doc.id, "text": doc.text,
            "metadata": json.dumps(doc.metadata), "category": doc.category,
        }])

    def add_docs(self, docs: list[KnowledgeDoc]):
        """Add multiple documents in batch."""
        self._lazy_init()
        texts = [d.text for d in docs]
        vecs = self._embed_many(texts)
        records = [
            {"vector": v, "id": d.id, "text": d.text,
             "metadata": json.dumps(d.metadata), "category": d.category}
            for v, d in zip(vecs, docs)
        ]
        self._table.add(records)

    def search(self, query: str, category: Optional[str] = None, top_k: int = 5) -> list[KnowledgeDoc]:
        """Search the knowledge base by semantic similarity."""
        self._lazy_init()
        vec = self._embed(query)
        results = self._table.search(vec).limit(top_k).to_list()
        docs = []
        for r in results:
            if category and r.get("category") != category:
                continue
            docs.append(KnowledgeDoc(
                id=r["id"], text=r["text"],
                metadata=json.loads(r.get("metadata", "{}")),
                category=r.get("category", ""),
            ))
        return docs

    def count(self) -> int:
        """Return total document count."""
        self._lazy_init()
        return self._table.count_rows()

    def seed_defaults(self, schema_info: Optional[list[KnowledgeDoc]] = None):
        """Seed the knowledge base with default schema info if empty."""
        self._lazy_init()
        if self.count() > 0:
            return
        if schema_info:
            self.add_docs(schema_info)