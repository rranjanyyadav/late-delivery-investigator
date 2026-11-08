"""Module 6: retrieval-augmented answers over the ops policy documents.

    build:  docs/policies/*.md --chunk by "## " section--> embed --> Chroma (data/chroma)
    ask:    question --embed--> top-k chunks --LLM, cite or refuse--> answer

    python -m ldi.rag build
    python -m ldi.rag ask "Who do we escalate to if a state misses the SLA twice?"
"""
import os
import re
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import chromadb

from ldi.config import CHROMA_DIR, POLICY_DIR
from ldi.llm import LLM

# Quiet the Hugging Face libraries (a symlink warning on Windows, progress bars).
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dims, ~90 MB, fast on CPU
COLLECTION = "policies"
# Retrieval finds the best sections, then the LLM reads the WHOLE policies they
# belong to (see expand_to_documents). In testing, compound questions ("targets
# AND escalation steps") ranked generic sections such as Purpose highly and
# pushed POL-01 §Targets to 6th-7th place, out of a section-only context.
TOP_K = 6
MAX_DOCS = 3  # policies are ~400 words, so at most ~1,200 words of context
# Cosine distance = 1 - cosine similarity. Chunks further away than this are
# treated as "not relevant", so an off-topic question retrieves nothing.
MAX_DISTANCE = 0.75
MAX_CHUNK_CHARS = 1200
NOT_COVERED = "The policy documents don't cover that."

ANSWER_PROMPT = f"""You answer questions about the company's operations policies.
- Use ONLY the policy excerpts given. No outside knowledge, no assumptions.
- Cite every rule you use like [POL-01 §Targets], using the ids and section names given.
- Quote thresholds, deadlines and owners exactly as written.
- If the excerpts answer only PART of the question, answer that part and say which
  part the excerpts do not cover.
- Only if the excerpts answer none of it, reply exactly: "{NOT_COVERED}"
- 2-5 sentences."""


@dataclass
class Chunk:
    id: str
    doc_id: str
    title: str
    section: str
    text: str
    distance: float | None = None

    @property
    def citation(self) -> str:
        return f"[{self.doc_id} §{self.section}]"


@dataclass
class RagResult:
    question: str
    status: str = "ok"  # ok | not_covered
    answer: str = ""
    chunks: list[Chunk] = field(default_factory=list)


# --------------------------------------------------------------- chunking
def parse_policy(path: Path) -> list[Chunk]:
    """One chunk per '## ' section. Each chunk starts with the policy title and
    section name, so a chunk such as 'Exceptions' still says WHICH policy's
    exceptions it is about when it is embedded and retrieved on its own."""
    text = re.sub(r"<!--.*?-->", "", path.read_text(encoding="utf-8"), flags=re.DOTALL)
    first_line = text.strip().splitlines()[0]
    match = re.match(r"#\s+(POL-\d+)\s+(.+)", first_line)
    if not match:
        raise ValueError(f"{path.name}: first line must be '# POL-XX Title', got {first_line!r}")
    doc_id, title = match.groups()

    chunks = []
    for section in re.split(r"(?m)^## ", text)[1:]:  # [0] is the title/metadata block
        heading, _, body = section.partition("\n")
        for i, piece in enumerate(_split_long(body.strip())):
            chunks.append(Chunk(
                id=f"{doc_id}-{len(chunks)}", doc_id=doc_id, title=title,
                section=heading.strip() + (f" (part {i + 1})" if i else ""),
                text=f"{doc_id} {title} - {heading.strip()}\n{piece}",
            ))
    return chunks


def _split_long(body: str) -> list[str]:
    """Keep sections whole; only split an over-long section at paragraph breaks."""
    pieces, current = [], ""
    for para in body.split("\n\n"):
        if current and len(current) + len(para) > MAX_CHUNK_CHARS:
            pieces.append(current.strip())
            current = ""
        current += para + "\n\n"
    if current.strip():
        pieces.append(current.strip())
    return pieces


def load_chunks(policy_dir: Path = POLICY_DIR) -> list[Chunk]:
    chunks = []
    for path in sorted(policy_dir.glob("*.md")):
        if not path.name.startswith("_"):  # _TEMPLATE.md is not a policy
            chunks.extend(parse_policy(path))
    return chunks


# ------------------------------------------------------------- embeddings
@lru_cache(maxsize=1)
def _embedder():
    # Imported lazily: loading torch takes a few seconds, and only RAG needs it.
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)


def embed(texts: list[str]) -> list[list[float]]:
    # normalize -> unit vectors, so cosine similarity is just a dot product
    return _embedder().encode(texts, normalize_embeddings=True).tolist()


# ------------------------------------------------------------------ index
def build_index(policy_dir: Path = POLICY_DIR, chroma_dir: Path = CHROMA_DIR) -> int:
    """(Re)build the Chroma collection from scratch. Returns the chunk count."""
    chunks = load_chunks(policy_dir)
    if not chunks:
        raise RuntimeError(f"No policy files found in {policy_dir}")
    client = chromadb.PersistentClient(path=str(chroma_dir))
    if COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION)  # rebuild, so edited/deleted docs never linger
    collection = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    collection.add(
        ids=[c.id for c in chunks],
        documents=[c.text for c in chunks],
        embeddings=embed([c.text for c in chunks]),
        metadatas=[{"doc_id": c.doc_id, "title": c.title, "section": c.section} for c in chunks],
    )
    return len(chunks)


@lru_cache(maxsize=1)
def _collection():
    return chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(COLLECTION)


def retrieve(question: str, k: int = TOP_K) -> list[Chunk]:
    """Top-k chunks closer than MAX_DISTANCE, nearest first."""
    res = _collection().query(query_embeddings=embed([question]), n_results=k)
    chunks = []
    for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0],
                                    res["metadatas"][0], res["distances"][0]):
        if dist <= MAX_DISTANCE:
            chunks.append(Chunk(cid, meta["doc_id"], meta["title"], meta["section"], doc, dist))
    return chunks


def expand_to_documents(hits: list[Chunk], max_docs: int = MAX_DOCS) -> list[Chunk]:
    """Parent-document retrieval: chunks are for FINDING, documents are for
    ANSWERING. Take the policies of the best-matching chunks (up to max_docs,
    best first) and return every section of them, in document order. A
    compound question then still sees e.g. §Targets even if that section
    ranked 7th on its own. Each chunk keeps its distance if it was a hit."""
    doc_ids = list(dict.fromkeys(c.doc_id for c in hits))[:max_docs]
    distances = {c.id: c.distance for c in hits}
    res = _collection().get(where={"doc_id": {"$in": doc_ids}},
                            include=["documents", "metadatas"])
    chunks = [Chunk(cid, m["doc_id"], m["title"], m["section"], doc, distances.get(cid))
              for cid, doc, m in zip(res["ids"], res["documents"], res["metadatas"])]
    # Group by document (best-matching first), sections in their original order.
    return sorted(chunks, key=lambda c: (doc_ids.index(c.doc_id), int(c.id.rsplit("-", 1)[1])))


def policy_titles() -> list[str]:
    """'POL-01 Delivery SLA and On-Time Targets', ... (used by the router prompt)."""
    metas = _collection().get(include=["metadatas"])["metadatas"]
    return sorted({f"{m['doc_id']} {m['title']}" for m in metas})


# ----------------------------------------------------------------- answer
def answer(question: str, llm: LLM | None = None) -> RagResult:
    hits = retrieve(question)
    result = RagResult(question=question)
    if not hits:
        # Nothing relevant retrieved: refuse WITHOUT calling the LLM, so it has
        # no chance to answer from its own general knowledge.
        result.status, result.answer = "not_covered", NOT_COVERED
        return result
    result.chunks = expand_to_documents(hits)
    context = "\n\n".join(f"{c.citation}\n{c.text}" for c in result.chunks)
    result.answer = (llm or LLM()).chat([
        {"role": "system", "content": ANSWER_PROMPT},
        {"role": "user", "content": f"Policy excerpts:\n{context}\n\nQuestion: {question}"},
    ])
    if result.answer.strip().strip('"') == NOT_COVERED:  # whole answer, not a partial one
        result.status = "not_covered"
    return result


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if len(sys.argv) >= 2 and sys.argv[1] == "build":
        print(f"Indexed {build_index()} chunks from {POLICY_DIR}")
    elif len(sys.argv) >= 3 and sys.argv[1] == "ask":
        r = answer(" ".join(sys.argv[2:]))
        for c in r.chunks:
            hit = f"{c.distance:.3f}" if c.distance is not None else "  -  "
            print(f"  {hit}  {c.citation}")
        print(f"\n{r.answer}")
    else:
        sys.exit('Usage: python -m ldi.rag build | python -m ldi.rag ask "question"')
