"""
Agent 2 — RAG Knowledge Agent
Vector Store: FAISS + sentence-transformers (all-MiniLM-L6-v2)
Knowledge: DermNet conditions + clinical research facts
Retrieval: MMR (Maximal Marginal Relevance) for diversity
"""

import os
import logging
import json
from typing import Optional
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger("dermai.rag_agent")

FAISS_INDEX_PATH = "/tmp/dermai_faiss_index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings():
    """
    Load embeddings — tries sentence-transformers, falls back to OpenAI.
    """
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        logger.info(f"Loaded HuggingFace embeddings: {EMBED_MODEL}")
        return embeddings, "huggingface"
    except Exception as e:
        logger.warning(f"HuggingFace embeddings failed ({e}), trying OpenAI")

    try:
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            base_url=os.environ.get("OPENAI_API_BASE"),
            model="text-embedding-3-small"
        )
        logger.info("Loaded OpenAI embeddings: text-embedding-3-small")
        return embeddings, "openai"
    except Exception as e:
        logger.warning(f"OpenAI embeddings failed ({e}), using keyword fallback")

    return None, "keyword_fallback"


def build_documents_from_knowledge() -> list[Document]:
    """
    Converts DermNet knowledge base to LangChain Documents with rich metadata.
    Each condition gets multiple chunked documents for granular retrieval.
    """
    from knowledge.dermnet_knowledge import DERMNET_CONDITIONS, SKINCARE_RESEARCH_FACTS

    documents = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " "]
    )

    for condition in DERMNET_CONDITIONS:
        base_meta = {
            "condition_id": condition["id"],
            "condition_name": condition["name"],
            "category": condition["category"],
            "icd_code": condition.get("icd_code", ""),
            "keywords": ", ".join(condition.get("keywords", [])),
            "source": "DermNet Knowledge Base"
        }

        overview_text = f"""
CONDITION: {condition['name']} (ICD: {condition.get('icd_code', 'N/A')})
CATEGORY: {condition['category']}

DESCRIPTION:
{condition['description']}

SYMPTOMS: {', '.join(condition.get('symptoms', []))}

SEVERITY: {', '.join(condition.get('severity_levels', [condition.get('severity', 'variable')]))}
""".strip()
        documents.append(Document(
            page_content=overview_text,
            metadata={**base_meta, "chunk_type": "overview"}
        ))

        treatments = condition.get("treatments", {})
        if treatments:
            treat_text = f"""
TREATMENT PROTOCOLS for {condition['name']}:

"""
            if isinstance(treatments, dict):
                for category, items in treatments.items():
                    if isinstance(items, list):
                        treat_text += f"{category.upper().replace('_', ' ')}: {', '.join(items)}\n"
                    elif isinstance(items, str):
                        treat_text += f"{category.upper().replace('_', ' ')}: {items}\n"
            elif isinstance(treatments, list):
                treat_text += '\n'.join(f"- {t}" for t in treatments)

            pathophysiology = condition.get("pathophysiology", "")
            if pathophysiology:
                treat_text += f"\nPATHOPHYSIOLOGY:\n{pathophysiology}"

            documents.append(Document(
                page_content=treat_text.strip(),
                metadata={**base_meta, "chunk_type": "treatment"}
            ))

        routine = condition.get("routine", {})
        if routine:
            routine_text = f"SKINCARE ROUTINE for {condition['name']}:\n\n"

            for time_of_day in ["morning", "evening", "weekly"]:
                steps = routine.get(time_of_day, [])
                if steps:
                    routine_text += f"{time_of_day.upper()} ROUTINE:\n"
                    for s in steps:
                        routine_text += f"  Step {s['order']}. {s['step']}: {s['product']}\n"
                        routine_text += f"    Active: {s.get('active', 'N/A')}\n"
                        routine_text += f"    How: {s['instruction']}\n"
                    routine_text += "\n"

            lifestyle = condition.get("lifestyle", condition.get("triggers_to_avoid", []))
            if lifestyle:
                routine_text += f"LIFESTYLE TIPS: {'; '.join(lifestyle)}\n"

            ingr_avoid = condition.get("ingredients_to_avoid", [])
            if ingr_avoid:
                routine_text += f"INGREDIENTS TO AVOID: {', '.join(ingr_avoid)}\n"

            documents.append(Document(
                page_content=routine_text.strip(),
                metadata={**base_meta, "chunk_type": "routine"}
            ))

        chunks = splitter.split_documents(
            [Document(page_content=overview_text, metadata=base_meta)]
        )
        for chunk in chunks:
            chunk.metadata["chunk_type"] = "semantic_chunk"
            documents.append(chunk)

    for fact in SKINCARE_RESEARCH_FACTS:
        documents.append(Document(
            page_content=f"RESEARCH FACT — {fact['fact']}:\n{fact['content']}\nSource: {fact['source']}",
            metadata={
                "condition_name": "General Research",
                "chunk_type": "research",
                "source": fact["source"]
            }
        ))

    logger.info(f"Built {len(documents)} documents from knowledge base")
    return documents


class KeywordFallbackRetriever:
    """
    Fallback when vector embeddings unavailable.
    Uses TF-IDF-style keyword scoring.
    """

    def __init__(self, documents: list[Document]):
        self.documents = documents

    def score(self, doc: Document, query_terms: list[str]) -> float:
        content_lower = doc.page_content.lower()
        meta_text = " ".join(str(v) for v in doc.metadata.values()).lower()
        full_text = content_lower + " " + meta_text
        score = 0
        for term in query_terms:
            term_lower = term.lower()
            count = full_text.count(term_lower)
            score += count * (2 if len(term) > 6 else 1)
        return score

    def get_relevant_documents(self, query: str, k: int = 5) -> list[Document]:
        terms = [t for t in query.lower().split() if len(t) > 3]
        scored = [(doc, self.score(doc, terms)) for doc in self.documents]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored[:k] if score > 0]


class RAGAgent:
    """
    Agent 2: RAG Knowledge Retrieval
    - Builds FAISS vector store from DermNet knowledge
    - Uses MMR retrieval for diverse, relevant results
    - Augments prompts with retrieved clinical context
    - Falls back to keyword search if embeddings unavailable
    """

    def __init__(self):
        self.name = "RAGAgent"
        self.vectorstore: Optional[FAISS] = None
        self.embeddings = None
        self.embed_type = None
        self.fallback_retriever: Optional[KeywordFallbackRetriever] = None
        self.documents: list[Document] = []

    async def initialize(self):
        """Build or load FAISS index at startup."""
        logger.info("RAGAgent initializing...")
        self.documents = build_documents_from_knowledge()

        self.embeddings, self.embed_type = get_embeddings()

        if self.embeddings:
            try:
                import os
                if os.path.exists(f"{FAISS_INDEX_PATH}/index.faiss"):
                    logger.info("Loading existing FAISS index")
                    self.vectorstore = FAISS.load_local(
                        FAISS_INDEX_PATH,
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                else:
                    logger.info(f"Building FAISS index from {len(self.documents)} documents")
                    self.vectorstore = FAISS.from_documents(self.documents, self.embeddings)
                    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
                    self.vectorstore.save_local(FAISS_INDEX_PATH)
                    logger.info("FAISS index saved")

            except Exception as e:
                logger.error(f"FAISS initialization failed: {e}")
                self.vectorstore = None

        if not self.vectorstore:
            logger.warning("Using keyword fallback retriever")
            self.fallback_retriever = KeywordFallbackRetriever(self.documents)
            self.embed_type = "keyword_fallback"

        logger.info(f"RAGAgent ready | embed_type={self.embed_type} | docs={len(self.documents)}")

    def retrieve(self, query: str, k: int = 5, use_mmr: bool = True) -> list[Document]:
        """
        Retrieve relevant documents for a query.
        MMR (Maximal Marginal Relevance): balances relevance + diversity.
        """
        if self.vectorstore:
            try:
                if use_mmr:
                    docs = self.vectorstore.max_marginal_relevance_search(
                        query, k=k, fetch_k=k * 3, lambda_mult=0.6
                    )
                else:
                    docs = self.vectorstore.similarity_search(query, k=k)
                return docs
            except Exception as e:
                logger.error(f"FAISS retrieval error: {e}")

        if self.fallback_retriever:
            return self.fallback_retriever.get_relevant_documents(query, k=k)

        return []

    def retrieve_for_conditions(self, conditions: list[str]) -> list[Document]:
        """
        Retrieve knowledge specifically for detected skin conditions.
        Multi-query approach: one query per condition + combined query.
        """
        all_docs = []
        seen_contents = set()

        queries = conditions[:3] + [" ".join(conditions[:3])]

        for query in queries:
            docs = self.retrieve(query, k=3)
            for doc in docs:
                content_key = doc.page_content[:100]
                if content_key not in seen_contents:
                    seen_contents.add(content_key)
                    all_docs.append(doc)

        logger.info(f"Retrieved {len(all_docs)} unique docs for conditions: {conditions}")
        return all_docs[:8]

    def format_context(self, documents: list[Document], max_chars: int = 3000) -> str:
        """
        Format retrieved documents into LLM-ready context string.
        Includes source metadata for citation.
        """
        if not documents:
            return "No specific knowledge retrieved. Apply general evidence-based dermatology principles."

        context_parts = []
        total_chars = 0

        for i, doc in enumerate(documents):
            meta = doc.metadata
            condition = meta.get("condition_name", "General")
            chunk_type = meta.get("chunk_type", "info")
            source = meta.get("source", "DermNet")

            section = f"[SOURCE {i+1}: {condition} | {chunk_type} | {source}]\n{doc.page_content}\n"

            if total_chars + len(section) > max_chars:
                remaining = max_chars - total_chars
                if remaining > 200:
                    context_parts.append(section[:remaining] + "...[truncated]")
                break

            context_parts.append(section)
            total_chars += len(section)

        return "\n---\n".join(context_parts)

    def get_sources_metadata(self, documents: list[Document]) -> list[dict]:
        """Extract source metadata for API response (for citation display)."""
        sources = []
        seen = set()
        for doc in documents:
            name = doc.metadata.get("condition_name", "General")
            if name not in seen:
                seen.add(name)
                sources.append({
                    "condition": name,
                    "category": doc.metadata.get("category", ""),
                    "icd_code": doc.metadata.get("icd_code", ""),
                    "chunk_type": doc.metadata.get("chunk_type", ""),
                    "source": doc.metadata.get("source", "DermNet Knowledge Base")
                })
        return sources


rag_agent = RAGAgent()
