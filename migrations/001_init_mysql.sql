-- Initial MySQL schema for sessions, events, RAG, alternatives, knowledge graph, procurement

CREATE TABLE IF NOT EXISTS sessions (
  session_id VARCHAR(64) NOT NULL,
  user_id VARCHAR(255) NOT NULL,
  channel VARCHAR(64) NOT NULL,
  state VARCHAR(64) NOT NULL DEFAULT 'idle',
  started_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  last_event_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
  PRIMARY KEY (session_id),
  KEY ix_sessions_user_channel (user_id, channel)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS events (
  event_id BIGINT NOT NULL AUTO_INCREMENT,
  ts DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  trace_id VARCHAR(128) NOT NULL,
  session_id VARCHAR(64) NOT NULL,
  user_id VARCHAR(255) NULL,
  event_type VARCHAR(64) NOT NULL,
  payload JSON NOT NULL,
  latency_ms INT NULL,
  error JSON NULL,
  PRIMARY KEY (event_id),
  KEY ix_events_trace (trace_id),
  KEY ix_events_session (session_id),
  KEY ix_events_type (event_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS sales_sessions (
  session_id VARCHAR(64) NOT NULL,
  state JSON NOT NULL,
  updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
  PRIMARY KEY (session_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS documents (
  doc_id VARCHAR(255) NOT NULL,
  text MEDIUMTEXT NOT NULL,
  source VARCHAR(255) NULL,
  source_type VARCHAR(64) NULL,
  client_id VARCHAR(64) NULL,
  product_id VARCHAR(64) NULL,
  lang VARCHAR(8) NULL,
  metadata JSON NOT NULL,
  embedding JSON NULL,
  embedding_model VARCHAR(64) NULL,
  ingested_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  PRIMARY KEY (doc_id),
  KEY ix_documents_client (client_id),
  KEY ix_documents_product (product_id),
  KEY ix_documents_source_type (source_type),
  KEY ix_documents_lang (lang)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS product_vectors (
  product_id VARCHAR(128) NOT NULL,
  vector JSON NOT NULL,
  metadata JSON NOT NULL,
  species VARCHAR(64) NULL,
  grade VARCHAR(32) NULL,
  price DECIMAL(12,2) NULL,
  in_stock TINYINT(1) NOT NULL DEFAULT 1,
  dimensions JSON NULL,
  updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
  PRIMARY KEY (product_id),
  KEY ix_product_vectors_species (species),
  KEY ix_product_vectors_grade (grade),
  KEY ix_product_vectors_in_stock (in_stock)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS kg_entities (
  entity_id VARCHAR(128) NOT NULL,
  entity_type VARCHAR(32) NOT NULL,
  attributes JSON NOT NULL,
  updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
  PRIMARY KEY (entity_id),
  KEY ix_kg_entity_type (entity_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS kg_edges (
  edge_id BIGINT NOT NULL AUTO_INCREMENT,
  src_id VARCHAR(128) NOT NULL,
  dst_id VARCHAR(128) NOT NULL,
  relation VARCHAR(64) NOT NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  PRIMARY KEY (edge_id),
  KEY ix_kg_edges_src_rel (src_id, relation),
  UNIQUE KEY uq_kg_edge (src_id, dst_id, relation),
  CONSTRAINT fk_kg_edges_src FOREIGN KEY (src_id) REFERENCES kg_entities (entity_id),
  CONSTRAINT fk_kg_edges_dst FOREIGN KEY (dst_id) REFERENCES kg_entities (entity_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Procurement module
CREATE TABLE IF NOT EXISTS rfq_specs (
  id INT NOT NULL AUTO_INCREMENT,
  spec JSON NOT NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  PRIMARY KEY (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS rfq_requests (
  id INT NOT NULL AUTO_INCREMENT,
  spec_id INT NOT NULL,
  vendor_id INT NOT NULL,
  status VARCHAR(16) NOT NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  PRIMARY KEY (id),
  KEY ix_rfq_spec_id (spec_id),
  CONSTRAINT fk_rfq_spec FOREIGN KEY (spec_id) REFERENCES rfq_specs (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS rfq_offers (
  id INT NOT NULL AUTO_INCREMENT,
  rfq_id INT NOT NULL,
  price_per_unit DECIMAL(12,2) NULL,
  min_batch DECIMAL(12,2) NULL,
  lead_time_days INT NULL,
  terms_text TEXT NOT NULL,
  vendor_score DECIMAL(5,2) NULL,
  raw_text TEXT NOT NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  PRIMARY KEY (id),
  KEY ix_rfq_offers_rfq (rfq_id),
  CONSTRAINT fk_rfq_offer FOREIGN KEY (rfq_id) REFERENCES rfq_requests (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
