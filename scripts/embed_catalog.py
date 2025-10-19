import os
import csv
import argparse
import math
import warnings
from io import BytesIO
from typing import List, Dict, Tuple, Optional


from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # handle partial images

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from tqdm import tqdm


import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import open_clip

# --------- Column names (match your schema exactly) ----------
COL_ID     = "unique_id"
COL_TD     = "title and description"
COL_IMGURL = "imgUrl"
COL_PURL   = "productURL"
COL_STARS  = "stars"
COL_PRICE  = "price"
COL_LPRICE = "listPrice"
COL_BEST   = "isBestSeller"
COL_BLM    = "boughtInLastMonth"

# --------- Models & device ----------
TEXT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384-d
CLIP_MODEL_NAME = "ViT-B-32"                                # 512-d
CLIP_PRETRAINED = "openai"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------- Utilities ----------
def vec_to_literal(np_vec: np.ndarray) -> str:
    """Format as pgvector input literal: [x1,x2,...] with ~6 decimals."""
    return "[" + ",".join(f"{float(x):.6f}" for x in np_vec.tolist()) + "]"

def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

def make_session(total_retries: int = 3, backoff: float = 0.5, pool: int = 32) -> requests.Session:
    sess = requests.Session()
    retry = Retry(
        total=total_retries,
        connect=total_retries,
        read=total_retries,
        status=total_retries,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=pool, pool_maxsize=pool)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    return sess

def load_image_from_url(session: requests.Session, url: str, timeout: int = 12) -> Optional[Image.Image]:
    try:
        r = session.get(url, timeout=timeout)
        r.raise_for_status()
        return Image.open(BytesIO(r.content)).convert("RGB")
    except Exception:
        return None

def count_rows(csv_path: str) -> int:
    # Fast-ish line count minus header
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        n = sum(1 for _ in f)
    return max(0, n - 1)

# --------- Batch processing ----------
def encode_text_batch(text_model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    # SentenceTransformers already handles batching internally with batch_size kwarg,
    # but we chunk explicitly to control memory if desired.
    return text_model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False
    ).astype(np.float32)

def encode_image_batch(
    clip_model,
    clip_preprocess,
    image_urls: List[str],
    session: requests.Session,
    device: str = DEVICE,
    amp: bool = True
) -> np.ndarray:
    """Download+preprocess images, then CLIP-encode in one mini-batch on GPU."""
    tensors = []
    for url in image_urls:
        if not url:
            tensors.append(None)
            continue
        img = load_image_from_url(session, url)
        if img is None:
            tensors.append(None)
        else:
            try:
                tensors.append(clip_preprocess(img))
            except Exception:
                tensors.append(None)

    # Stack valid tensors; keep an index map to scatter back
    idx_map = [i for i, t in enumerate(tensors) if t is not None]
    if idx_map:
        batch = torch.stack([tensors[i] for i in idx_map], dim=0).to(device)
    else:
        # No valid images; return zeros
        return np.zeros((len(image_urls), 512), dtype=np.float32)

    with torch.no_grad():
        if device == "cuda" and amp:
            # mixed precision for speed
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                feats = clip_model.encode_image(batch)
        else:
            feats = clip_model.encode_image(batch)

        feats = F.normalize(feats, p=2, dim=-1)  # [N,512]
        feats_np = feats.detach().cpu().numpy().astype(np.float32)

    # Scatter back into full array, fill missing with zeros
    out = np.zeros((len(image_urls), feats_np.shape[1]), dtype=np.float32)
    for j, i in enumerate(idx_map):
        out[i] = feats_np[j]
    return out

def process_batch(
    rows: List[Dict[str, str]],
    text_model: SentenceTransformer,
    clip_model,
    clip_preprocess,
    session: requests.Session,
) -> Tuple[List[str], List[str]]:
    """Return (text_literals, image_literals) aligned to input rows."""
    texts = [(r.get(COL_TD) or "").strip() for r in rows]
    img_urls = [(r.get(COL_IMGURL) or "").strip() for r in rows]

    # Text
    text_vecs = encode_text_batch(text_model, texts)  # [B,384]
    text_lits = [vec_to_literal(v) for v in text_vecs]

    # Image
    img_vecs = encode_image_batch(clip_model, clip_preprocess, img_urls, session)  # [B,512]
    img_lits = [vec_to_literal(v) for v in img_vecs]

    return text_lits, img_lits

# --------- Main ----------
def main():
    parser = argparse.ArgumentParser(
        description="Fast text+image embeddings for a product catalog CSV with progress."
    )
    parser.add_argument("--in",  dest="in_csv",  required=True, help="Path to input catalog CSV")
    parser.add_argument("--out", dest="out_csv", required=True, help="Path to output CSV with embeddings")
    parser.add_argument("--batch-size", type=int, default=128, help="Rows per mini-batch (default: 128)")
    parser.add_argument("--http-pool", type=int, default=32, help="HTTP connection pool size (default: 32)")
    parser.add_argument("--http-retries", type=int, default=3, help="HTTP retry count (default: 3)")
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA mixed precision for CLIP")
    args = parser.parse_args()

    in_csv  = args.in_csv
    out_csv = args.out_csv
    bs      = max(1, args.batch_size)

    if not os.path.isfile(in_csv):
        raise FileNotFoundError(f"Input CSV not found: {in_csv}")

    ensure_parent_dir(out_csv)

    print(f"Using device: {DEVICE}")
    print(f"Reading : {in_csv}")
    print(f"Writing : {out_csv}")

    # Models
    text_model = SentenceTransformer(TEXT_MODEL_NAME, device=DEVICE)
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED, device=DEVICE
    )
    clip_model.eval().to(DEVICE)

    session = make_session(total_retries=args.http_retries, pool=args.http_pool)

    total_rows = count_rows(in_csv)

    # Open files and stream in mini-batches; write as we go
    with open(in_csv, newline="", encoding="utf-8") as fin, \
         open(out_csv, "w", newline="", encoding="utf-8") as fout:

        reader = csv.DictReader(fin)
        required = {COL_ID, COL_TD, COL_IMGURL}
        missing = [c for c in required if c not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"Input CSV is missing required columns: {missing}. "
                f"Found columns: {reader.fieldnames}"
            )

        fieldnames = reader.fieldnames + ["text_embedding", "image_embedding"]
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        pbar = tqdm(total=total_rows or None, unit="row", ncols=100, desc="Embedding")
        buffer: List[Dict[str, str]] = []

        try:
            for row in reader:
                uid = (row.get(COL_ID) or "").strip()
                if not uid:
                    continue
                buffer.append(row)

                if len(buffer) >= bs:
                    text_lits, img_lits = process_batch(
                        buffer, text_model, clip_model, clip_preprocess, session
                    )
                    for r, t, im in zip(buffer, text_lits, img_lits):
                        r["text_embedding"]  = t
                        r["image_embedding"] = im
                        writer.writerow(r)
                    pbar.update(len(buffer))
                    buffer.clear()

            # Flush remainder
            if buffer:
                text_lits, img_lits = process_batch(
                    buffer, text_model, clip_model, clip_preprocess, session
                )
                for r, t, im in zip(buffer, text_lits, img_lits):
                    r["text_embedding"]  = t
                    r["image_embedding"] = im
                    writer.writerow(r)
                pbar.update(len(buffer))

        finally:
            pbar.close()

    print(f"✅ Done. Wrote: {out_csv}")

if __name__ == "__main__":
    # Silence PIL EXIF warnings etc.
    warnings.filterwarnings("ignore", category=UserWarning, module="PIL")
    main()