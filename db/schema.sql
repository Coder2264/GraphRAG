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
-- Seed: domain entity types (edit as needed for your domain)
-- ----------------------------------------------------------------
INSERT INTO graph_entity_types (name, description) VALUES
  ('Person',        'A human individual mentioned in the text'),
  ('Organization',  'A formal group, institution, or non-profit'),
  ('Location',      'A place, region, country, or geographic entity'),
  ('Event',         'A happening, occurrence, or incident with a time and place'),
  ('Product',       'A software product, tool, hardware device, or service offering'),
  ('Concept',       'An abstract idea, theory, methodology, or domain term'),
  ('ResearchPaper', 'An academic publication, preprint, or technical report'),
  ('Company',       'A commercial business or corporation'),
  ('Disease',       'A medical condition, illness, disorder, or syndrome'),
  ('Chemical',      'A chemical compound, element, drug, or molecular entity')
ON CONFLICT (name) DO NOTHING;

-- ----------------------------------------------------------------
-- Seed: canonical relation types.
-- If an extracted relation does not match any of these types the
-- LLM must use RAW_RELATION (see extraction prompt) so no
-- information is silently discarded.
-- ----------------------------------------------------------------
INSERT INTO graph_relation_types (name, description) VALUES
  ('WORKS_AT',          'A person works at or is employed by an organization or company'),
  ('STUDIED_AT',        'A person studied or received a degree at an institution'),
  ('LOCATED_IN',        'An entity is physically located in or belongs to a place'),
  ('PART_OF',           'An entity is a component, member, or subset of another entity'),
  ('CEO_OF',            'A person is the chief executive officer of a company or organization'),
  ('COLLABORATES_WITH', 'Two entities work together or have a partnership'),
  ('OWNS',              'An entity owns, holds, or controls another entity'),
  ('FOUNDED_BY',        'An organization or project was founded or created by a person'),
  ('REPORTS_TO',        'A person or unit reports to another person or unit in a hierarchy'),
  ('HEADQUARTERED_IN',  'A company or organization is headquartered in a location')
ON CONFLICT (name) DO NOTHING;
