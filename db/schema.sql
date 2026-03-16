-- =============================================================================
-- RAG Chunks Schema
-- PostgreSQL + pgvector
--
-- Apply manually:
--   psql $POSTGRES_DSN -f db/schema.sql
--
-- NOTE: The embedding dimension (1024) must match EMBEDDING_DIM in .env.
--       nomic-embed-text → 768  |  mxbai-embed-large → 1024  |  all-minilm → 384
--       If you change models, drop and recreate the table.
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS rag_chunks (
    -- Chunk identity
    chunk_id      TEXT        PRIMARY KEY,               -- "{doc_id}__chunk_{i}"
    doc_id        TEXT        NOT NULL,                  -- parent document UUID
    chunk_index   INTEGER     NOT NULL,
    total_chunks  INTEGER     NOT NULL,

    -- Document provenance
    source        TEXT        NOT NULL DEFAULT '',       -- filename or URL

    -- Content + vector
    content       TEXT        NOT NULL DEFAULT '',
    embedding     vector(1024),                           -- change dim to match your model

    -- Housekeeping
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT rag_chunks_doc_chunk_unique UNIQUE (doc_id, chunk_index)
);

-- B-tree index for fast doc_id lookups (used by delete and filter queries)
CREATE INDEX IF NOT EXISTS rag_chunks_doc_id_idx
    ON rag_chunks (doc_id);

-- IVFFlat approximate nearest-neighbour index (cosine distance)
-- Tune `lists` after you have data: a good starting point is sqrt(row_count).
-- With < ~1 000 rows exact scan (no index) is typically faster anyway.
CREATE INDEX IF NOT EXISTS rag_chunks_embedding_idx
    ON rag_chunks
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- =============================================================================
-- GraphRAG: Configurable entity and relation type catalogs
-- These rows are loaded at extraction time and injected into the LLM prompt
-- so the extracted graph stays anchored to your domain vocabulary.
-- Edit / extend freely — no code changes required.
-- =============================================================================

CREATE TABLE IF NOT EXISTS graph_entity_types (
    id          SERIAL      PRIMARY KEY,
    name        TEXT        NOT NULL UNIQUE,          -- e.g. "Person"
    description TEXT        NOT NULL DEFAULT '',      -- shown verbatim to the LLM
    is_active   BOOLEAN     NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS graph_relation_types (
    id          SERIAL      PRIMARY KEY,
    name        TEXT        NOT NULL UNIQUE,          -- e.g. "WORKS_FOR"
    description TEXT        NOT NULL DEFAULT '',
    is_active   BOOLEAN     NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ----------------------------------------------------------------
-- Seed: 5 domain-relevant examples (edit as needed for your domain)
-- ----------------------------------------------------------------
INSERT INTO graph_entity_types (name, description) VALUES
  ('Person',       'A human individual mentioned in the text'),
  ('Organization', 'A company, institution, or formal group'),
  ('Regulation',   'A law, rule, policy, or compliance requirement'),
  ('Product',      'A software product, tool, or service'),
  ('Location',     'A place, region, country, or geographic entity')
ON CONFLICT (name) DO NOTHING;

INSERT INTO graph_relation_types (name, description) VALUES
  ('WORKS_FOR',    'A person is employed by or works for an organization'),
  ('GOVERNS',      'A regulation governs or applies to an entity'),
  ('LOCATED_IN',   'An entity is located in or associated with a place'),
  ('PRODUCES',     'An organization produces or develops a product'),
  ('RELATED_TO',   'A generic catch-all relationship between two entities')
ON CONFLICT (name) DO NOTHING;
