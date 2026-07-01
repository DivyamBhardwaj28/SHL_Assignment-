import json
import logging
from pathlib import Path
import faiss
import torch
from sentence_transformers import SentenceTransformer

# Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_PATH = Path("shl_product_catalog.json")
INDEX_PATH = Path("shl_catalog.index")
METADATA_PATH = Path("shl_metadata.json")

def enrich_item(item):
    """Dynamically adds Metadata Enrichment fields."""
    name = item.get("name", "").lower()
    
    # Defaults
    family = "General"
    aliases = [item.get("name")]
    skills = []
    
    # Enrichment Logic
    if "opq" in name:
        family = "OPQ"
        aliases.extend(["OPQ32r", "Personality Test", "Leadership Report", "Competency Report"])
        skills.extend(["Leadership", "Management", "Competency"])
    elif "excel" in name or "word" in name:
        family = "Microsoft Office"
        skills.extend(["Office", "Spreadsheet", "Documentation"])
    elif "verify" in name:
        family = "Verify"
        aliases.append("Aptitude Test")
        
    return family, aliases, skills

def build_vector_store():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        catalog_data = json.load(f)

    texts_to_embed = []
    metadata = []
    
    for idx, item in enumerate(catalog_data):
        # 1. Enrich
        family, aliases, skills = enrich_item(item)
        
        # 2. Update item object (ensure ID is never null)
        item["faiss_id"] = idx 
        item["family"] = family
        item["aliases"] = aliases
        item["skills"] = skills
        
        # 3. Create rich text blob for embedding
        # Prefix required for BGE models
        combined_text = (
            f"Represent this assessment for retrieval: {item.get('name')}. "
            f"Family: {family}. "
            f"Aliases: {', '.join(aliases)}. "
            f"Skills: {', '.join(skills)}. "
            f"Description: {item.get('description', '')}."
        )
        texts_to_embed.append(combined_text)
        metadata.append(item)

    # 4. Use BGE model (Upgrade from MiniLM)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Note: Ensure you have sentence-transformers installed
    model = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)
    
    logger.info("Generating BGE embeddings...")
    embeddings = model.encode(
        texts_to_embed, 
        normalize_embeddings=True, 
        convert_to_numpy=True
    )

    # 5. Build Index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    # 6. Save
    faiss.write_index(index, str(INDEX_PATH))
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    
    logger.info("Index and Enriched Metadata saved.")

if __name__ == "__main__":
    build_vector_store()