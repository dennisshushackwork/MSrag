-- ======================================================
-- Create needed extensions
-- ======================================================
CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;
CREATE EXTENSION IF NOT EXISTS pg_trgm CASCADE;
-- ======================================================
-- Base Tables
-- ======================================================
-- Documents # Needs some metadata for later
CREATE TABLE IF NOT EXISTS Document (
    document_id   SERIAL PRIMARY KEY,
    content       TEXT NOT NULL,
    content_tsv   tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    created_at    TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Chunks
CREATE TABLE IF NOT EXISTS Chunk (
    chunk_id          SERIAL PRIMARY KEY,
    chunk_document_id INT NOT NULL REFERENCES Document(document_id),
    chunk_text        TEXT NOT NULL,
    chunk_text_tsv    tsvector GENERATED ALWAYS AS (to_tsvector('english', chunk_text)) STORED,
    chunk_tokens      INTEGER,
    chunk_type        TEXT,
    start_index       INTEGER,
    end_index         INTEGER,
    chunk_emb         VECTOR(256),
    chunk_embed       BOOLEAN DEFAULT FALSE
);

-- Entities
CREATE TABLE IF NOT EXISTS Entity (
    entity_id        SERIAL PRIMARY KEY,
    entity_name      TEXT UNIQUE NOT NULL,
    entity_name_tsv  tsvector GENERATED ALWAYS AS (to_tsvector('english', entity_name)) STORED,
    entity_emb       VECTOR(256),
    entity_embed     BOOLEAN DEFAULT FALSE
);

-- Relationships
CREATE TABLE IF NOT EXISTS Relationship (
    rel_id              SERIAL PRIMARY KEY,
    from_entity         INT NOT NULL REFERENCES Entity(entity_id),
    to_entity           INT NOT NULL REFERENCES Entity(entity_id),
    rel_description     TEXT,
    rel_description_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', rel_description)) STORED,
    rel_emb             VECTOR(256),
    rel_tokens          INTEGER,
    rel_summary         TEXT,
    rel_chunk_id        INT NOT NULL REFERENCES Chunk(chunk_id),
    rel_document_id     INT NOT NULL REFERENCES Document(document_id),
    rel_weight          INTEGER,
    rel_embed           BOOLEAN DEFAULT FALSE
);

-- Communities
CREATE TABLE IF NOT EXISTS CommunityGroup (
    community_id           SERIAL PRIMARY KEY,
    community_level        INT,
    community_parent       INT,
    community_num_nodes    INT,
    community_summary      TEXT,
    community_tokens       INT,
    community_summary_tsv  tsvector GENERATED ALWAYS AS (to_tsvector('english', community_summary)) STORED,
    community_emb          VECTOR(256),
    community_embed        BOOLEAN DEFAULT FALSE
);

-- ======================================================
-- Association (Join) Tables
-- ======================================================

-- Entities ↔ Documents
CREATE TABLE IF NOT EXISTS EntityDocument (
    entity_id   INT NOT NULL REFERENCES Entity(entity_id),
    document_id INT NOT NULL REFERENCES Document(document_id),
    PRIMARY KEY (entity_id, document_id)
);

-- Entities ↔ Chunks
CREATE TABLE IF NOT EXISTS EntityChunk (
    entity_id INT NOT NULL REFERENCES Entity(entity_id),
    chunk_id  INT NOT NULL REFERENCES Chunk(chunk_id),
    PRIMARY KEY (entity_id, chunk_id)
);

-- Communities ↔ Entities
CREATE TABLE IF NOT EXISTS CommunityNode (
    community_id INT NOT NULL REFERENCES CommunityGroup(community_id),
    entity_id    INT NOT NULL REFERENCES Entity(entity_id),
    PRIMARY KEY (community_id, entity_id)
);

-- Communities ↔ Documents
CREATE TABLE IF NOT EXISTS CommunityDocument (
    community_id INT NOT NULL REFERENCES CommunityGroup(community_id),
    document_id  INT NOT NULL REFERENCES Document(document_id),
    PRIMARY KEY (community_id, document_id)
);

-- Communities ↔ Chunks
CREATE TABLE IF NOT EXISTS CommunityChunk (
    community_id INT NOT NULL REFERENCES CommunityGroup(community_id),
    chunk_id     INT NOT NULL REFERENCES Chunk(chunk_id),
    PRIMARY KEY (community_id, chunk_id)
);

-- ============================================================================================================
-- Entity Similarity Table (keeps the records of all entities who have cosine similarity above 0.85
-- ============================================================================================================
CREATE TABLE IF NOT EXISTS EntitySimilarity(
    from_entity INT NOT NULL REFERENCES Entity(entity_id),
    to_entity INT NOT NULL REFERENCES Entity(entity_id),
    similarity FLOAT NOT NULL,
    PRIMARY KEY (from_entity, to_entity)
);

-- ======================================================
-- ChromaDB Evaluation
-- ======================================================
-- Add evaluation metadata table for storing evaluation runs
CREATE TABLE IF NOT EXISTS EvaluationRun (
    run_id SERIAL PRIMARY KEY,
    run_name TEXT,
    chunk_type TEXT,
    chunk_size INTEGER,
    chunk_overlap INTEGER,
    iou_mean FLOAT,
    precision_mean FLOAT,
    recall_mean FLOAT,
    precision_omega_mean FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    metadata JSONB
);

-- ======================================================
-- Indexes
-- ======================================================
-- Flags & types
CREATE INDEX IF NOT EXISTS idx_chunk_type        ON Chunk(chunk_type);
CREATE INDEX IF NOT EXISTS idx_chunk_embed_flag  ON Chunk(chunk_embed);
CREATE INDEX IF NOT EXISTS idx_entity_embed_flag ON Entity(entity_embed);
CREATE INDEX IF NOT EXISTS idx_community_level   ON CommunityGroup(community_level);

-- Join-table lookups
CREATE INDEX IF NOT EXISTS idx_ed_document_id ON EntityDocument(document_id);
CREATE INDEX IF NOT EXISTS idx_ec_chunk_id    ON EntityChunk(chunk_id);
CREATE INDEX IF NOT EXISTS idx_cn_entity_id   ON CommunityNode(entity_id);
CREATE INDEX IF NOT EXISTS idx_cd_document_id ON CommunityDocument(document_id);
CREATE INDEX IF NOT EXISTS idx_cc_chunk_id    ON CommunityChunk(chunk_id);
CREATE INDEX IF NOT EXISTS idx_from_entity_sim ON EntitySimilarity(from_entity);
CREATE INDEX IF NOT EXISTS idx_to_entity_sim ON EntitySimilarity(to_entity);

-- Composite indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_relationship_entities    ON Relationship(from_entity, to_entity);

-- Relationship foreign-key lookups
CREATE INDEX IF NOT EXISTS idx_relationship_from_entity ON Relationship(from_entity);
CREATE INDEX IF NOT EXISTS idx_relationship_to_entity   ON Relationship(to_entity);
CREATE INDEX IF NOT EXISTS idx_relationship_chunk_id    ON Relationship(rel_chunk_id);
CREATE INDEX IF NOT EXISTS idx_relationship_document_id ON Relationship(rel_document_id);

-- Full-text search indexes
CREATE INDEX IF NOT EXISTS idx_document_content_tsv    ON Document       USING gin(content_tsv);
CREATE INDEX IF NOT EXISTS idx_chunk_text_tsv          ON Chunk          USING gin(chunk_text_tsv);
CREATE INDEX IF NOT EXISTS idx_entity_name_tsv         ON Entity         USING gin(entity_name_tsv);
CREATE INDEX IF NOT EXISTS idx_rel_description_tsv     ON Relationship   USING gin(rel_description_tsv);
CREATE INDEX IF NOT EXISTS idx_community_summary_tsv   ON CommunityGroup USING gin(community_summary_tsv);

-- Partial indexes for better performance on embedded items
CREATE INDEX IF NOT EXISTS idx_chunk_embedded_only     ON Chunk(chunk_id) WHERE chunk_embed = true;
CREATE INDEX IF NOT EXISTS idx_entity_embedded_only    ON Entity(entity_id) WHERE entity_embed = true;
CREATE INDEX IF NOT EXISTS idx_rel_embedded_only       ON Relationship(rel_id) WHERE rel_embed = true;
CREATE INDEX IF NOT EXISTS idx_community_embedded_only ON CommunityGroup(community_id) WHERE community_embed = true;

-- Create GiST trigram index - better for distance queries and ORDER BY operations
CREATE INDEX IF NOT EXISTS idx_entity_name_gist_trgm
ON Entity USING gist (entity_name gist_trgm_ops);

-- Vector-similarity (DiskANN)
CREATE INDEX IF NOT EXISTS idx_chunk_emb      ON Chunk          USING diskann (chunk_emb vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_entity_emb     ON Entity         USING diskann (entity_emb vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_rel_emb        ON Relationship   USING diskann (rel_emb vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_community_emb  ON CommunityGroup USING diskann (community_emb vector_cosine_ops);

-- Index for Chroma:
CREATE INDEX IF NOT EXISTS idx_chunk_positions ON Chunk(start_index, end_index);
CREATE INDEX IF NOT EXISTS idx_chunk_type_positions ON Chunk(chunk_type, start_index, end_index);
CREATE INDEX IF NOT EXISTS idx_evaluation_run_type ON EvaluationRun(chunk_type, chunk_size, chunk_overlap);

