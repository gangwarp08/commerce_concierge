CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS public.catalog_items (
  unique_id TEXT PRIMARY KEY,
  title_desc TEXT,
  img_url TEXT,
  product_url TEXT,
  stars FLOAT,
  price FLOAT,
  listPrice FLOAT,
  isBestSeller BOOLEAN,
  boughtInLastMonth INT,
  text_embedding vector(384),
  image_embedding vector(512)
);

-- Example: copy data from CSV after embedding (psql)
-- \copy public.catalog_items FROM 'data/embedded_catalog.csv' CSV HEADER;

-- Recommended indexes for cosine similarity
CREATE INDEX IF NOT EXISTS idx_text_embedding  
  ON public.catalog_items USING ivfflat (text_embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_image_embedding 
  ON public.catalog_items USING ivfflat (image_embedding vector_cosine_ops);