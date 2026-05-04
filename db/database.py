"""
Database module — SQLAlchemy with psycopg2 (sync) driver.
Wraps sync operations in thread pool for async compatibility.
"""
import os
import logging
from sqlalchemy import create_engine, Column, String, Integer, Float, Text, DateTime, JSON, event
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session
from sqlalchemy.sql import func
from contextlib import contextmanager

logger = logging.getLogger("dermai.db")

DATABASE_URL = os.environ.get("DATABASE_URL", "")
# Normalize URL for psycopg2 sync driver
if DATABASE_URL.startswith("postgresql+asyncpg://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://", 1)
elif DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)


class Base(DeclarativeBase):
    pass


class DiagnosisModel(Base):
    __tablename__ = "dermai_diagnoses"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, nullable=False, index=True)
    image_url = Column(Text)
    conditions = Column(JSON, default=list)
    skin_type = Column(String)
    overall_score = Column(Float, default=0)
    confidence_scores = Column(JSON, default=dict)
    summary = Column(Text)
    detailed_analysis = Column(Text)
    recommendations = Column(JSON, default=list)
    rag_sources = Column(JSON, default=list)
    routine = Column(JSON, default=dict)
    severity = Column(String, default="mild")
    urgency = Column(String, default="routine")
    vision_source = Column(String, default="unknown")
    embed_type = Column(String, default="keyword_fallback")
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ChatMessageModel(Base):
    __tablename__ = "dermai_chat"

    id = Column(Integer, primary_key=True, autoincrement=True)
    diagnosis_id = Column(Integer, nullable=False, index=True)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    sources = Column(JSON, default=list)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=5) if DATABASE_URL else None
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine) if engine else None


def init_db():
    if not engine:
        logger.warning("No DATABASE_URL — running without persistence")
        return
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialized")


@contextmanager
def get_db_session():
    """Context manager for database sessions."""
    if not SessionLocal:
        yield None
        return
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


async def get_db():
    """FastAPI dependency — yields sync session."""
    if not SessionLocal:
        yield None
        return
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
