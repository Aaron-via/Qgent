CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE papers (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    authors JSONB,
    abstract TEXT,
    domain TEXT,
    method_tags JSONB,
    content_tree JSONB,
    status TEXT DEFAULT 'pending',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE experiments (
    exp_id SERIAL PRIMARY KEY,
    paper_id INT REFERENCES papers(id) ON DELETE CASCADE,
    exp_group_id UUID,
    task TEXT,
    dataset TEXT,
    model_name TEXT,
    metric TEXT,
    value FLOAT,
    hyperparams JSONB,
    is_ours BOOLEAN,
    domain_specific JSONB,
    evidence_path JSONB
);

CREATE INDEX idx_exp_group ON experiments(exp_group_id);
CREATE INDEX idx_exp_metric ON experiments(metric);
CREATE INDEX idx_exp_model ON experiments(model_name);

CREATE TABLE claims (
    claim_id SERIAL PRIMARY KEY,
    paper_id INT REFERENCES papers(id) ON DELETE CASCADE,
    claim_type TEXT CHECK (claim_type IN ('contribution', 'limitation', 'assumption', 'position')),
    content TEXT NOT NULL,
    embedding VECTOR(1024),
    domain_specific JSONB,
    evidence_path JSONB
);
