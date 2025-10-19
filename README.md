# commerce_concierge
<<<<<<< HEAD
Your personal shopping assistant. Drop in a text description or attach an image [file/doc/URL] and I will help you get the best for you.
=======
Commerce Concierge — AI-Powered Product Recommendation Agent

Your personal shopping assistant. Drop in a text description or attach an image [file/doc/URL] and I will help you get the best for you.

Link to access from here - https://chatgpt.com/g/g-68f14f0b1160819196231b901064226e-commerce-concierge


**GitHub Repo Structure**
``` bash
commerce-concierge/
├── README.md
├── api/
│   ├── main.py                  # FastAPI microservice for embedding + query
│   ├── requirements.txt         # Python dependencies
│   └── Dockerfile               # Optional containerization
├── workflows/
│   ├── n8n_commerce_concierge.json   # Your exported n8n workflow
│   └── openapi_schema.json      # Custom GPT → n8n Webhook schema
├── data/
│   ├── catalog_sample.csv       # Sample of embedded product catalog
│   └── neon_connection.sql      # Example pgvector schema setup
└── scripts/
    └── embed_catalog.py         # Your provided embedding script (text + image)
```

### 📘 README.md (Documentation)
**Overview**

Commerce Concierge is an AI shopping assistant that integrates:
1. A Custom GPT interface for natural conversation.
2. An n8n workflow backend to process text or image queries.
3. A PostgreSQL + pgvector catalog hosted on Neon.tech for similarity search.
Users can chat, upload an image, or describe a product, and the agent recommends the top-5 most relevant catalog items based on semantic embeddings.


### TECHNOLOGY STACK AND DESIGN DECISIONS
**Component**	            **Technology**	                **Reason**
Frontend (Chat)	            OpenAI Custom GPT	            Conversational UX without hosting overhead
Middleware	                n8n Workflow	                Low-code orchestration of OpenAI + CLIP + Postgres
Embeddings (Text)	        text-embedding-3-small(OpenAI)	Fast, compact 384-d embeddings
Embeddings (Image)	        openai/clip-vit-base-patch32	Industry-standard 512-d image vectors
Database	                Neon Postgres + pgvector	    Cloud-native vector similarity search
ETL Pipeline	            Python(SentenceTransformers + OpenCLIP)	Efficient batch embedding (GPU optional)



### SYSTEM ARCHITECTURE
``` sql
User → Custom GPT → n8n Webhook → OpenAI / HuggingFace Embeddings
                                  ↓
                              pgvector Query → Top-5 Products
                                  ↓
                            n8n → GPT Response → User
```


### WORKFLOW LOGIC

1. Webhook Trigger → Receives JSON payload from the Custom GPT:
``` json
{
  "inputText": "black leather backpack",
  "intent": "text_rec"
}
```

2. Switch Node routes intent:
- text_rec → OpenAI Embeddings → pgvector query by text
- image_rec → HF CLIP Embeddings → pgvector query by image
- general_talk → GPT-4o mini for small-talk replies

3. pgvector Conversion node normalizes vectors and formats as SQL literal.

4. PostgreSQL Query retrieves top-5 items:
``` sql
SELECT title_desc, product_url, price
FROM public.catalog_items
ORDER BY text_embedding <=> '{{ $json.vector }}'::vector
LIMIT 5;
```

5. Respond to Webhook → Sends product recommendations back to GPT.




### API ENDPOINTS

POST /api/query

Field	        Type	        Description
inputText	    string	        User text input
intent	        string	        text_rec, image_rec, or general_talk
file	        string (URI)	Optional image URL for image_rec


Response:
``` json
{
  "results": [
    {"title": "Backpack, Black Leather", "price": 79.99, "similarity": 0.92},
    ...
  ]
}
```

### pgvector SCHEMA SETUP

``` sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE catalog_items (
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
```


### EMBEDDING PIPELINE
Your script (embed_catalog.py) encodes both text and images, writing back embeddings as pgvector literals.

Usage:
``` bash
python embed_catalog.py --in data/catalog.csv --out data/embedded_catalog.csv
```

Performance Notes
- Uses SentenceTransformer for 384-dim text vectors.
- Uses open_clip for 512-dim image vectors.
- GPU acceleration + mixed precision (AMP) supported.
- Retry logic + progress bar via tqdm.


### DEPLOYMENT
Option 1: Local (for testing)
- Run n8n locally (npm run dev or Docker)
- Import n8n_commerce_concierge.json
- Connect PostgreSQL Neon and OpenAI/HF credentials
- Deploy Custom GPT with your webhook endpoint

Option 2: Hosted (recommended)
- Host n8n on n8n.cloud
- Use Neon.tech (serverless Postgres)
- Expose Webhook URL publicly for GPT integration
- Optionally wrap n8n in a FastAPI layer (api/main.py) for modular REST access


### SAMPLE FASTAPI SERVICE

```python

from fastapi import FastAPI, UploadFile
import requests, os

app = FastAPI()

@app.post("/api/query")
async def query(inputText: str = None, imageUrl: str = None):
    payload = {"inputText": inputText, "intent": "text_rec" if inputText else "image_rec", "file": imageUrl}
    r = requests.post("https://pgangwar.app.n8n.cloud/webhook/e33d0db3-e6a3-4da4-b04b-064ffe596724", json=payload)
    return r.json()
```
>>>>>>> 1265c4e (Initial clean commit)
