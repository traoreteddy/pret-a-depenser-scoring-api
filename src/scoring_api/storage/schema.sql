-- Schéma de stockage des données de production (idempotent, appliqué au démarrage de l'API).

CREATE TABLE IF NOT EXISTS predictions (
    id                   BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    request_id           UUID NOT NULL UNIQUE,
    ts                   TIMESTAMPTZ NOT NULL DEFAULT now(),
    model_name           TEXT NOT NULL,
    model_version        TEXT NOT NULL,
    model_backend        TEXT NOT NULL,
    threshold            REAL NOT NULL,
    proba_defaut         DOUBLE PRECISION,
    decision             TEXT CHECK (decision IN ('Accordé', 'Refusé')),
    latency_total_ms     REAL,
    latency_features_ms  REAL,
    latency_inference_ms REAL,
    status_code          SMALLINT NOT NULL DEFAULT 200,
    error_type           TEXT,
    error_message        TEXT,
    sampled              BOOLEAN NOT NULL DEFAULT false,
    raw_input            JSONB,        -- champs bruts reçus (si sampled)
    features             JSONB,        -- 38 variables envoyées au modèle (si sampled)
    client_ref           TEXT,         -- en-tête X-Client-Ref (jointure avec les labels différés)
    scenario             TEXT          -- en-tête X-Scenario (trafic simulé)
);
CREATE INDEX IF NOT EXISTS predictions_ts_idx         ON predictions (ts);
CREATE INDEX IF NOT EXISTS predictions_decision_idx   ON predictions (decision, ts);
CREATE INDEX IF NOT EXISTS predictions_sampled_idx    ON predictions (ts) WHERE sampled;
CREATE INDEX IF NOT EXISTS predictions_error_idx      ON predictions (ts) WHERE error_type IS NOT NULL;
CREATE INDEX IF NOT EXISTS predictions_client_ref_idx ON predictions (client_ref);
CREATE INDEX IF NOT EXISTS predictions_scenario_idx   ON predictions (scenario, ts);

CREATE TABLE IF NOT EXISTS api_errors (
    id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    request_id  UUID NOT NULL,
    ts          TIMESTAMPTZ NOT NULL DEFAULT now(),
    path        TEXT NOT NULL,
    status_code SMALLINT NOT NULL,
    error_type  TEXT NOT NULL,         -- validation_error | model_error | internal_error
    detail      JSONB,                 -- erreurs Pydantic (loc, msg, type)
    raw_body    JSONB,                 -- corps reçu (pour rejouer / analyser)
    scenario    TEXT
);
CREATE INDEX IF NOT EXISTS api_errors_ts_idx ON api_errors (ts);

CREATE TABLE IF NOT EXISTS prediction_labels (
    request_id UUID PRIMARY KEY REFERENCES predictions (request_id) ON DELETE CASCADE,
    target     SMALLINT NOT NULL CHECK (target IN (0, 1)),
    labeled_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    source     TEXT NOT NULL DEFAULT 'simulation'
);
