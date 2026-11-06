# build_faiss_enhanced.py — Build FAISS index from multiple data sources
# Handles PDFs, TXT files, and extensionless text files in data/raw/.

import csv
import faiss, pickle, re
import os
import sys
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── Configuration ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.resolve()

DATA_DIR = ROOT / "data" / "raw"
DATA_SOURCES = [DATA_DIR]
GITA_CSV = DATA_DIR / "Bhagwad_Gita.csv"

OUTPUT_DIR   = ROOT / "faiss_index"
EMBED_MODEL  = "all-mpnet-base-v2"
CHUNK_SIZE   = 1200   # Characters per chunk
CHUNK_OVERLAP = 220   # Overlap between chunks

# ── Friendly Source Labels ────────────────────────────────────────────────────
SOURCE_LABELS = {
    # PDFs in data/raw/
    "atomic habits": "Atomic Habits (Clear)",
    "daniel goleman": "Emotional Intelligence (Goleman)",
    "dhammapada": "The Dhammapada",
    "feeling good": "Feeling Good (Burns)",
    "marcus-aurelius": "Meditations (Marcus Aurelius)",
    "marcus aurelius": "Meditations (Marcus Aurelius)",
    "tao te ching": "Tao Te Ching",
    "taoofseneca_vol1": "Tao of Seneca Vol. 1",
    "taoofseneca_vol2": "Tao of Seneca Vol. 2",
    "taoofseneca_vol3": "Tao of Seneca Vol. 3",
    "the body keeps": "The Body Keeps the Score (van der Kolk)",
    "the happiness trap": "The Happiness Trap (Harris)",
    "the power of now": "The Power of Now (Tolle)",
    "thinking fast": "Thinking, Fast and Slow (Kahneman)",
    "thinking, fast": "Thinking, Fast and Slow (Kahneman)",
    "bhagwad_gita": "Bhagavad Gita",
    "bhagavad_gita": "Bhagavad Gita",
    "bhagwad gita": "Bhagavad Gita",
    "bhagavad gita": "Bhagavad Gita",
    
    # Bhagavad Gita chapters
    "chapter_1": "Bhagavad Gita (Chapter 1)",
    "chapter_2": "Bhagavad Gita (Chapter 2)",
    "chapter_3": "Bhagavad Gita (Chapter 3)",
    "chapter_4": "Bhagavad Gita (Chapter 4)",
    "chapter_5": "Bhagavad Gita (Chapter 5)",
    "chapter_6": "Bhagavad Gita (Chapter 6)",
    "chapter_7": "Bhagavad Gita (Chapter 7)",
    "chapter_8": "Bhagavad Gita (Chapter 8)",
    "chapter_9": "Bhagavad Gita (Chapter 9)",
    "chapter_10": "Bhagavad Gita (Chapter 10)",
    "chapter_11": "Bhagavad Gita (Chapter 11)",
    "chapter_12": "Bhagavad Gita (Chapter 12)",
    "chapter_13": "Bhagavad Gita (Chapter 13)",
    "chapter_14": "Bhagavad Gita (Chapter 14)",
    "chapter_15": "Bhagavad Gita (Chapter 15)",
    "chapter_16": "Bhagavad Gita (Chapter 16)",
    "chapter_17": "Bhagavad Gita (Chapter 17)",
    "chapter_18": "Bhagavad Gita (Chapter 18)",
    "gita": "Bhagavad Gita",
    "sample_gita": "Bhagavad Gita",
}

# ── Text cleaning ─────────────────────────────────────────────────────────────
_ZW = re.compile(r'[\u200b\ufeff\u200c\u200d]')

def clean_text(text: str) -> str:
    """Basic cleaning to remove zero-width chars and excessive whitespace"""
    if not isinstance(text, str):
        return ""
    s = _ZW.sub("", text)
    s = re.sub(r'-\s*\n\s*', '', s)
    s = re.sub(r'(?<=[a-z])\n(?=[a-z])', ' ', s)
    s = re.sub(r'\s{2,}', ' ', s)
    return s.strip()

# ── PDF extraction ────────────────────────────────────────────────────────────
def extract_pdf(path: Path) -> str:
    """Extract text from PDF using PyMuPDF"""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(path))
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()
        return "\n".join(text_parts)
    except Exception as e:
        print(f"   ⚠️  Failed to read PDF {path.name}: {e}")
        return ""

# ── Text file extraction ──────────────────────────────────────────────────────
def extract_txt(path: Path) -> str:
    """Extract text from .txt file with encoding fallbacks"""
    for encoding in ["utf-8", "latin-1", "cp1252"]:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    print(f"   ⚠️  Failed to decode {path.name} with common encodings")
    return ""

# ── File processing ───────────────────────────────────────────────────────────
def read_gita_csv(path: Path) -> list[dict]:
    """Load Bhagavad Gita CSV rows as structured verse records."""
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception as e:
        print(f"   Failed to read CSV {path.name}: {e}")
        return []

    required = {"Chapter", "Verse", "Shloka", "Transliteration", "EngMeaning"}
    if not rows or not required.issubset(rows[0].keys()):
        print(f"   Skipping unsupported CSV format: {path.name}")
        return []
    return rows

def gita_row_to_text(row: dict) -> str:
    """Format one CSV row as readable verse text for storage and retrieval."""
    chapter = (row.get("Chapter") or "").strip()
    verse = (row.get("Verse") or "").strip()
    shloka = clean_text(row.get("Shloka") or "")
    transliteration = clean_text(row.get("Transliteration") or "")
    translation = clean_text(row.get("EngMeaning") or "")
    word_meaning = clean_text(row.get("WordMeaning") or "")

    parts = [
        f"Bhagavad Gita {chapter}.{verse}",
        f"Shloka: {shloka}" if shloka else "",
        f"Transliteration: {transliteration}" if transliteration else "",
        f"Translation: {translation}" if translation else "",
        f"Word meaning and commentary: {word_meaning}" if word_meaning else "",
    ]
    return "\n".join(part for part in parts if part).strip()

def get_source_label(filename: str) -> str:
    """Get friendly label for a source file"""
    lower = filename.lower()
    for key, label in sorted(SOURCE_LABELS.items(), key=lambda item: len(item[0]), reverse=True):
        if key in lower:
            return label
    # Fallback: prettify filename
    return filename.replace("_", " ").replace("-", " ").title()

def process_file(path: Path) -> tuple[str, str]:
    """
    Extract text from file and return (text, label)
    Returns ("", "") if extraction fails
    """
    if not path.is_file():
        return "", ""
    
    # Get text based on extension
    if path.suffix.lower() == ".pdf":
        text = extract_pdf(path)
    elif path.suffix.lower() in [".txt", ".text", ""]:
        text = extract_txt(path)
    else:
        print(f"   ⚠️  Skipping unsupported file type: {path.name}")
        return "", ""
    
    if not text or len(text) < 100:
        return "", ""
    
    # Clean and get label
    cleaned = clean_text(text)
    label = get_source_label(path.stem)
    
    return cleaned, label

# ── Chunking ──────────────────────────────────────────────────────────────────
def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks, preferring paragraph/sentence breaks."""
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        target_end = min(start + chunk_size, text_len)
        end = target_end
        
        # Try to break at sentence boundary
        if target_end < text_len:
            last_segment = text[max(start, target_end - 240):target_end]
            for delimiter in ["\n\n", ". ", "? ", "! ", "; ", "\n"]:
                split_pos = last_segment.rfind(delimiter)
                if split_pos >= 0:
                    end = max(start + 1, target_end - len(last_segment) + split_pos + len(delimiter))
                    break

        chunk = text[start:end]
        
        if chunk.strip():
            chunks.append(chunk.strip())
        
        start = max(end - overlap, start + 1)
    
    return chunks

# ── Main indexing ─────────────────────────────────────────────────────────────
def build_index():
    """Build FAISS index from all source directories"""
    
    print("="*70)
    print("🪷  Building KrishnaAI Knowledge Base")
    print("="*70)
    
    # Load embedding model
    print(f"\n📦 Loading embedding model: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)
    print("✓ Model loaded\n")
    
    # Collect all files from all sources
    all_files = []
    for source_dir in DATA_SOURCES:
        if not source_dir.exists():
            print(f"⚠️  Directory not found: {source_dir}")
            continue
        
        print(f"📂 Scanning: {source_dir}")
        files = list(source_dir.glob("*"))
        files = [f for f in files if f.is_file() and not f.name.startswith(".")]
        print(f"   Found {len(files)} files")
        all_files.extend(files)
    
    if not all_files:
        print("\nNo files found! Please add books to data/raw/")
        return
    
    print(f"\n📚 Processing {len(all_files)} total files...\n")
    
    # Process all files
    all_chunks = []
    embedding_texts = []
    all_meta = []
    
    for i, filepath in enumerate(all_files, 1):
        print(f"[{i}/{len(all_files)}] {filepath.name}")

        if filepath.suffix.lower() == ".csv":
            rows = read_gita_csv(filepath)
            if not rows:
                print(f"   âš ï¸  Skipped (no CSV verses extracted)\n")
                continue

            label = get_source_label(filepath.stem)
            print(f"   âœ“ {len(rows)} verses | Label: {label}")
            for row in rows:
                chapter = (row.get("Chapter") or "").strip()
                verse = (row.get("Verse") or "").strip()
                chunk = gita_row_to_text(row)
                if not chunk:
                    continue
                citation = f"Bhagavad Gita {chapter}.{verse}"
                all_chunks.append(chunk)
                embedding_texts.append(f"Source: {label}\nCitation: {citation}\n\n{chunk}")
                all_meta.append({
                    "source_file": filepath.name,
                    "source_label": label,
                    "source_path": str(filepath),
                    "chapter": chapter,
                    "verse": verse,
                    "citation": citation,
                    "kind": "gita_csv",
                })
            print()
            continue
        
        # Extract text
        text, label = process_file(filepath)
        if not text:
            print(f"   ⚠️  Skipped (no content extracted)\n")
            continue
        
        # Chunk the text
        chunks = chunk_text(text)
        print(f"   ✓ {len(chunks)} chunks | Label: {label}")
        
        # Store chunks and metadata
        for chunk in chunks:
            all_chunks.append(chunk)
            embedding_texts.append(f"Source: {label}\nTitle: {filepath.stem}\n\n{chunk}")
            all_meta.append({
                "source_file": filepath.name,
                "source_label": label,
                "source_path": str(filepath)
            })
        
        print()
    
    if not all_chunks:
        print("❌ No chunks created! Check your data files.")
        return
    
    print(f"📊 Total chunks to embed: {len(all_chunks)}")
    print(f"📖 Unique sources: {len(set(m['source_label'] for m in all_meta))}\n")
    
    # Embed all chunks
    print("🔄 Generating embeddings (this may take a minute)...")
    embeddings = embedder.encode(
        embedding_texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32
    )
    embeddings = np.array(embeddings, dtype="float32")
    print("✓ Embeddings generated\n")
    
    # Create FAISS index
    print("🔍 Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product = cosine similarity when normalized
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    print(f"✓ Index created ({index.ntotal} vectors, {dim} dimensions)\n")
    
    # Save everything
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    
    print("💾 Saving to disk...")
    faiss.write_index(index, str(OUTPUT_DIR / "index.faiss"))
    with open(OUTPUT_DIR / "chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)
    with open(OUTPUT_DIR / "meta.pkl", "wb") as f:
        pickle.dump(all_meta, f)
    
    print("✓ Saved to:", OUTPUT_DIR)
    print("\n" + "="*70)
    print("✅ DONE! Knowledge base ready.")
    print("="*70)
    print(f"\n📚 Sources indexed:")
    for label in sorted(set(m['source_label'] for m in all_meta)):
        print(f"   • {label}")
    print()

if __name__ == "__main__":
    build_index()
