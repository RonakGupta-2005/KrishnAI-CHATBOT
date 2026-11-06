# rag_terminal_enhanced.py — KrishnaAI with Memory & Warmth
# Major improvements: conversation history, natural tone, context awareness

import csv
import faiss, pickle, numpy as np, requests, re, time, json
from pathlib import Path
from collections import deque
from datetime import datetime

ROOT       = Path(__file__).parent.resolve()
FAISS_INDEX = str(ROOT / "faiss_index" / "index.faiss")
CHUNKS_PKL  = str(ROOT / "faiss_index" / "chunks.pkl")
META_PKL    = str(ROOT / "faiss_index" / "meta.pkl")
GITA_CSV = ROOT / "data" / "raw" / "Bhagwad_Gita.csv"

EMBED_MODEL    = "all-mpnet-base-v2"
OLLAMA_URL     = "http://localhost:11434/api/generate"
OLLAMA_MODEL   = "llama3:8b"
OLLAMA_TIMEOUT = 180
TOP_K          = 6
MIN_CONTEXT_SCORE = 0.32

# Conversation memory - stores last N exchanges
MAX_HISTORY = 6  # Keep last 6 exchanges (12 messages)

# ── Crisis keywords — always safe-gate ───────────────────────────────────────
CRISIS_PATTERNS = re.compile(
    r'\b(suicid\w*|self.harm|kill myself|end my life|want to die|no reason to live|hurt myself|don\'t want to live)\b',
    re.IGNORECASE
)

HIGH_RISK_PATTERNS = re.compile(
    r'\b('
    r'suicidal|suicide|kill myself|end my life|want to die|don\'t want to live|do not want to live|'
    r'going to die|going to kill myself|suicide plan|planned my suicide|'
    r'can\'t stay safe|cannot stay safe|not safe with myself|'
    r'took pills|overdose|hanging myself|hang myself|jump off|jumping off|'
    r'cut my wrist|cutting my wrist|hurt myself tonight|goodbye forever'
    r')\b',
    re.IGNORECASE
)

CRISIS_RESPONSE = (
    "I can hear that you're going through something really difficult right now. "
    "Please know that what you're feeling matters, and you don't have to face this alone.\n\n"
    "I really think it would help to talk to someone who can give you the support you deserve right now:\n\n"
    "• **iCall (India):** 9152987821\n"
    "• **Vandrevala Foundation (24/7):** 1860-2662-345\n"
    "• **International Crisis Centres:** https://www.iasp.info/resources/Crisis_Centres/\n\n"
    "Please reach out to them or someone you trust. You matter, and there are people who want to help."
)

# ── Cleaning ──────────────────────────────────────────────────────────────────
_ZW      = re.compile(r'[\u200b\ufeff\u200c\u200d]')
_EMAIL   = re.compile(r'\S+@\S+\.\S+')
_URL     = re.compile(r'(?:www\.|http:|https:|mailto:)')
_DIGITS  = re.compile(r'\d{4,}')
_FOOTER  = re.compile(r'(copyright|©|isbn|all rights reserved)', re.I)
_GITA_QUERY = re.compile(r"\b(bg|bhagavad\s*gita|bhagavad\s*geeta|bhagwad\s*gita|bhagwad\s*geeta|bhagvad\s*gita|bhagvad\s*geeta|gita|geeta)\b", re.I)
_CHAPTER_RE = re.compile(r"\bchapter\s+([a-z0-9-]+)\b", re.I)
_GITA_DOT_RE = re.compile(r"\b(?:bg|gita|geeta|bhagavad\s*gita|bhagavad\s*geeta|bhagwad\s*gita|bhagwad\s*geeta)\s+([0-9]+)\s*[.:/-]\s*([0-9]+)\b", re.I)
_VERSE_AFTER_RE = re.compile(r"\b(?:verse|text|sloka|shloka)\s+([a-z0-9-]+)\b", re.I)
_VERSE_BEFORE_RE = re.compile(r"\b([a-z0-9-]+)\s+(?:verse|text|sloka|shloka)\b", re.I)
_TEXT_SECTION_RE = re.compile(r"(?ms)^TEXT\s*\n\s*(\d+)\s*\n(.*?)(?=^TEXT\s*\n\s*\d+\s*\n|\Z)")
_GITA_CITATION_RE = re.compile(r"\bBhagavad\s+Gita\s+([0-9]+)\.([0-9]+)\b", re.I)
_FOLLOWUP_REFERENCE_RE = re.compile(
    r"\b(this|that|the|above|previous)\s+(verse|shloka|sloka|teaching|wisdom|message)\b|\b(it|this|that)\b",
    re.I,
)

ORDINAL_WORDS = {
    "first": 1, "1st": 1,
    "second": 2, "2nd": 2,
    "third": 3, "3rd": 3,
    "fourth": 4, "4th": 4,
    "fifth": 5, "5th": 5,
    "sixth": 6, "6th": 6,
    "seventh": 7, "7th": 7,
    "eighth": 8, "8th": 8,
    "ninth": 9, "9th": 9,
    "tenth": 10, "10th": 10,
    "eleventh": 11, "11th": 11,
    "twelfth": 12, "12th": 12,
    "thirteenth": 13, "13th": 13,
    "fourteenth": 14, "14th": 14,
    "fifteenth": 15, "15th": 15,
    "sixteenth": 16, "16th": 16,
    "seventeenth": 17, "17th": 17,
    "eighteenth": 18, "18th": 18,
}

DISPLAY_REPLACEMENTS = [
    ("å", "ri"),
    ("Å", "Ri"),
    ("ä", "a"),
    ("Ä", "A"),
    ("ö", "t"),
    ("Ö", "T"),
    ("ñ", "sh"),
    ("Ñ", "Sh"),
    ("ë", "n"),
    ("Ë", "N"),
    ("ò", "d"),
    ("Ò", "D"),
    ("ï", "n"),
    ("Ï", "N"),
    ("ç", "c"),
    ("Ç", "S"),
    ("é", "i"),
    ("É", "I"),
    ("ù", "h"),
    ("Ù", "H"),
    ("ü", "u"),
    ("Ü", "U"),
    ("à", "a"),
    ("À", "A"),
    ("á", "a"),
    ("Á", "A"),
    ("í", "i"),
    ("Í", "I"),
    ("ú", "u"),
    ("Ú", "U"),
    ("Š", ""),
    ("Œ", ""),
    ("œ", ""),
    ("™", "'"),
    ("ﬁ", "fi"),
    ("ﬂ", "fl"),
]

COMMON_WORD_FRAGMENTS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "he", "her", "his", "i", "in", "into", "is", "it", "my", "not", "of",
    "on", "or", "our", "she", "so", "than", "that", "the", "their", "them",
    "there", "these", "they", "this", "to", "was", "we", "were", "who", "with",
    "you", "your",
}

def fix_mojibake(text: str) -> str:
    if not isinstance(text, str):
        return ""
    if "Ã" not in text and "â" not in text:
        return text
    try:
        return text.encode("latin-1").decode("utf-8")
    except Exception:
        return text

def make_display_friendly(text: str) -> str:
    if not isinstance(text, str):
        return ""

    s = fix_mojibake(text)
    for old, new in DISPLAY_REPLACEMENTS:
        s = s.replace(old, new)

    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r" *([,.;:?!])", r"\1", s)
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def repair_wrapped_words(text: str) -> str:
    def repl(match):
        left = match.group(1)
        right = match.group(2)
        if left.lower() in COMMON_WORD_FRAGMENTS or right.lower() in COMMON_WORD_FRAGMENTS:
            return left + " " + right
        if len(left) <= 4 or len(right) <= 3:
            return left + right
        return left + " " + right

    return re.sub(r"\b([A-Za-z]+)\s*\n\s*([a-z]+)\b", repl, text)

def clean_text(text: str) -> str:
    if not isinstance(text, str): return ""
    s = make_display_friendly(_ZW.sub("'", text))
    s = repair_wrapped_words(s)
    lines = []
    for ln in s.splitlines():
        ln = ln.strip()
        if not ln: continue
        if _EMAIL.search(ln) or _URL.search(ln) or _FOOTER.search(ln): continue
        if _DIGITS.search(ln) and len(ln) < 60: continue
        if len(ln) < 4 and not ln.isalpha(): continue
        non_alpha = sum(1 for c in ln if not c.isalnum() and not c.isspace()) / max(1, len(ln))
        if non_alpha > 0.25: continue
        lines.append(ln)
    return re.sub(r'\s{2,}', ' ', ' '.join(lines)).strip()

# ── Load / embed / retrieve ───────────────────────────────────────────────────
def load_system():
    print("Loading FAISS index + knowledge base …")
    index  = faiss.read_index(FAISS_INDEX)
    with open(CHUNKS_PKL, "rb") as f: chunks = pickle.load(f)
    with open(META_PKL,   "rb") as f: meta   = pickle.load(f)
    from sentence_transformers import SentenceTransformer
    embed = SentenceTransformer(EMBED_MODEL, local_files_only=True)
    print(f"Loaded {index.ntotal} vectors | {len(chunks)} chunks from {len({m.get('source_label') for m in meta})} sources")
    return index, chunks, meta, embed

def embed_query(embedder, q: str):
    v = embedder.encode([q], normalize_embeddings=True)
    return np.array(v, dtype="float32")

def is_bhagavad_gita_query(question: str) -> bool:
    return bool(question and _GITA_QUERY.search(question))

def parse_reference_number(token: str | None) -> int | None:
    if not token:
        return None
    token = token.strip().lower().rstrip(".,?!")
    if token.isdigit():
        return int(token)
    return ORDINAL_WORDS.get(token)

def parse_gita_reference(question: str) -> tuple[int, int] | None:
    if not is_bhagavad_gita_query(question):
        return None
    dot_match = _GITA_DOT_RE.search(question)
    if dot_match:
        return int(dot_match.group(1)), int(dot_match.group(2))
    chapter_match = _CHAPTER_RE.search(question)
    chapter = parse_reference_number(chapter_match.group(1)) if chapter_match else None
    verse = None
    for verse_match in (_VERSE_AFTER_RE.search(question), _VERSE_BEFORE_RE.search(question)):
        if not verse_match:
            continue
        verse = parse_reference_number(verse_match.group(1))
        if verse:
            break
    if chapter and verse:
        return chapter, verse
    return None

def parse_gita_citation(text: str) -> tuple[int, int] | None:
    match = _GITA_CITATION_RE.search(text or "")
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))

def resolve_contextual_gita_reference(question: str, history: list) -> tuple[int, int] | None:
    explicit_ref = parse_gita_reference(question)
    if explicit_ref:
        return explicit_ref

    if not _FOLLOWUP_REFERENCE_RE.search(question or ""):
        return None

    for turn in reversed(history or []):
        ref = parse_gita_citation(turn.get("content", ""))
        if ref:
            return ref
    return None

def gita_reference_retrieval(chapter: int, verse: int) -> list[dict]:
    match = extract_gita_verse(chapter, verse)
    if not match:
        return []
    body = match["text"] or match["translation"]
    return [{
        "score": 1.0,
        "vector_score": 1.0,
        "keyword_score": 1.0,
        "text": body,
        "citation": match["citation"],
        "meta": {
            "source_label": match["source_label"],
            "source_path": match["source_path"],
        },
    }]

def read_gita_csv_rows() -> list[dict]:
    if not GITA_CSV.exists():
        return []
    with GITA_CSV.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        return list(csv.DictReader(f))

def clean_gita_csv_text(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def extract_gita_verse(chapter: int, verse: int) -> dict | None:
    for row in read_gita_csv_rows():
        try:
            row_chapter = int((row.get("Chapter") or "").strip())
            row_verse = int((row.get("Verse") or "").strip())
        except ValueError:
            continue
        if row_chapter != chapter or row_verse != verse:
            continue

        shloka = clean_gita_csv_text(row.get("Shloka") or "")
        transliteration = clean_gita_csv_text(row.get("Transliteration") or "")
        translation = clean_gita_csv_text(row.get("EngMeaning") or "")
        word_meaning = clean_gita_csv_text(row.get("WordMeaning") or "")
        body_parts = [
            f"Shloka:\n{shloka}" if shloka else "",
            f"Transliteration:\n{transliteration}" if transliteration else "",
            f"Translation:\n{translation}" if translation else "",
        ]
        body = "\n\n".join(part for part in body_parts if part)

        return {
            "chapter": chapter,
            "verse": verse,
            "citation": f"Bhagavad Gita {chapter}.{verse}",
            "source_label": f"Bhagavad Gita (Chapter {chapter})",
            "text": body,
            "translation": translation,
            "word_meaning": word_meaning,
            "source_path": str(GITA_CSV),
        }
    return None

def answer_exact_reference(question: str) -> dict | None:
    gita_ref = parse_gita_reference(question)
    if not gita_ref:
        return None

    chapter, verse = gita_ref
    if not (1 <= chapter <= 18 and verse >= 1):
        return {
            "answer": f"I could not find Bhagavad Gita {chapter}.{verse}. Please check the chapter and verse numbers.",
            "citation": f"Bhagavad Gita {chapter}.{verse}",
            "retrieved": [],
        }

    match = extract_gita_verse(chapter, verse)
    if not match:
        return {
            "answer": f"I could not find Bhagavad Gita {chapter}.{verse} in the local CSV dataset.",
            "citation": f"Bhagavad Gita {chapter}.{verse}",
            "retrieved": [],
        }

    body = match["text"] or match["translation"]
    return {
        "answer": f"{match['citation']}\n\n{body}",
        "citation": match["citation"],
        "retrieved": [{
            "score": 1.0,
            "text": body,
            "citation": match["citation"],
            "meta": {
                "source_label": match["source_label"],
                "source_path": match["source_path"],
            },
        }],
    }

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z']{2,}")
_STOPWORDS = {
    "about", "after", "again", "against", "all", "also", "and", "any", "are",
    "because", "been", "but", "can", "could", "does", "for", "from", "give",
    "had", "has", "have", "how", "into", "just", "like", "more", "not",
    "now", "only", "should", "tell", "than", "that", "the", "their", "them",
    "then", "there", "these", "this", "what", "when", "where", "which",
    "with", "would", "you", "your",
}

SOURCE_ALIASES = {
    "atomic habits": "atomic habits",
    "james clear": "atomic habits",
    "emotional intelligence": "emotional intelligence",
    "goleman": "emotional intelligence",
    "dhammapada": "dhammapada",
    "feeling good": "feeling good",
    "burns": "feeling good",
    "meditations": "meditations",
    "marcus": "meditations",
    "aurelius": "meditations",
    "tao te ching": "tao te ching",
    "seneca": "seneca",
    "body keeps the score": "body keeps the score",
    "trauma": "body keeps the score",
    "happiness trap": "happiness trap",
    "power of now": "power of now",
    "tolle": "power of now",
    "thinking fast and slow": "thinking, fast and slow",
    "kahneman": "thinking, fast and slow",
    "yoga sutras": "yoga sutras",
    "patanjali": "yoga sutras",
    "bhagavad gita": "bhagavad gita",
    "bhagavad geeta": "bhagavad gita",
    "gita": "bhagavad gita",
    "geeta": "bhagavad gita",
}

SOURCE_KEY_LABELS = {
    "atomic habits": "Atomic Habits",
    "emotional intelligence": "Emotional Intelligence",
    "dhammapada": "The Dhammapada",
    "feeling good": "Feeling Good",
    "meditations": "Meditations",
    "tao te ching": "Tao Te Ching",
    "seneca": "Tao of Seneca",
    "body keeps the score": "The Body Keeps the Score",
    "happiness trap": "The Happiness Trap",
    "power of now": "The Power of Now",
    "thinking, fast and slow": "Thinking, Fast and Slow",
    "bhagavad gita": "Bhagavad Gita",
}

def query_terms(question: str) -> set[str]:
    return {
        token.lower().strip("'")
        for token in _TOKEN_RE.findall(question or "")
        if token.lower().strip("'") not in _STOPWORDS
    }

def lexical_overlap_score(question_terms: set[str], text: str) -> float:
    if not question_terms or not text:
        return 0.0
    text_terms = set(_TOKEN_RE.findall(text.lower()))
    return len(question_terms & text_terms) / max(4, len(question_terms))

def requested_source_key(question: str) -> str | None:
    q = (question or "").lower()
    for alias, key in SOURCE_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", q):
            return key
    return None

def source_key_for_label(label: str) -> str | None:
    lower_label = (label or "").lower()
    for key in SOURCE_KEY_LABELS:
        if key in lower_label:
            return key
    return None

def label_matches_source(label: str, source_key: str | None) -> bool:
    if not source_key:
        return True
    return source_key_for_label(label) == source_key

def source_key_from_text(text: str) -> str | None:
    explicit = requested_source_key(text)
    if explicit:
        return explicit

    citation_ref = parse_gita_citation(text)
    if citation_ref:
        return "bhagavad gita"

    return None

def resolve_active_source(question: str, history: list) -> str | None:
    explicit = requested_source_key(question)
    if explicit:
        return explicit

    if not _FOLLOWUP_REFERENCE_RE.search(question or ""):
        return None

    for turn in reversed(history or []):
        source_key = source_key_from_text(turn.get("content", ""))
        if source_key:
            return source_key
    return None

def source_display_name(source_key: str | None) -> str:
    return SOURCE_KEY_LABELS.get(source_key or "", source_key or "the selected source")

def retrieve(index, chunks, meta, q_emb, question: str = "", top_k=TOP_K, source_key: str | None = None):
    faiss.normalize_L2(q_emb)
    source_key = source_key or requested_source_key(question)
    search_k = index.ntotal if source_key else max(top_k * 20, top_k)
    D, I = index.search(q_emb, search_k)
    results = []
    terms = query_terms(question)
    seen = set()
    for score, idx in zip(D[0], I[0]):
        raw     = chunks[idx]
        cleaned = clean_text(raw)
        m       = meta[idx]
        label   = m.get("source_label", m.get("source_file", "Unknown Source"))
        if not label_matches_source(label, source_key):
            continue
        fingerprint = re.sub(r"\W+", " ", cleaned.lower())[:220]
        if not cleaned or fingerprint in seen:
            continue
        seen.add(fingerprint)
        lexical = lexical_overlap_score(terms, cleaned)
        source_bonus = 0.08 if source_key and source_key in label.lower() else 0.0
        final_score = float(score) + (0.12 * lexical) + source_bonus
        results.append({
            "score":    final_score,
            "vector_score": float(score),
            "keyword_score": lexical,
            "text":     cleaned,
            "meta":     m,
            "citation": label,
        })
    results.sort(key=lambda r: r["score"], reverse=True)
    if is_bhagavad_gita_query(question):
        gita_results = [r for r in results if "bhagavad gita" in r["citation"].lower()]
        other_results = [r for r in results if "bhagavad gita" not in r["citation"].lower()]
        results = gita_results + other_results
    return results[:top_k]

# ── Enhanced Persona with natural warmth ──────────────────────────────────────
PERSONA = """You are KrishnaAI, a warm and caring companion for mental wellness conversations.

YOUR CORE NATURE:
- You're like a wise, caring friend who listens deeply and responds naturally
- You have warmth, empathy, and genuine presence in every exchange
- You remember what was said earlier in the conversation and build on it
- You match the user's energy and tone - casual when they're casual, deeper when they need depth

CONVERSATION PRINCIPLES:

1. GREETINGS & SMALL TALK:
   When someone says "hi", "hello", "hey" - just greet them warmly back (1 sentence).
   Examples: "Hey! How are you doing today?" or "Hi there! What's on your mind?"
   DO NOT: Launch into therapy-speak or ask profound questions unprompted.

2. REMEMBER THE CONVERSATION:
   You can see the recent chat history. Reference it naturally:
   - "You mentioned earlier that you had a headache..."
   - "I remember you were feeling stuck yesterday..."
   - "It sounds like this is still weighing on you..."
   Build continuity. Show you're truly listening and remember.

3. MATCH THEIR ENERGY:
   - Casual question → casual, brief answer
   - Deep sharing → compassionate, thoughtful response
   - Practical problem → practical solutions
   - Emotional pain → gentle, supportive presence

4. BE CONVERSATIONAL:
   Talk like a real person would. Use natural phrasing:
   - "That sounds really tough" not "I hear that you're experiencing difficulty"
   - "Have you tried..." not "I would recommend that you consider..."
   - "Yeah, that makes sense" not "Your feelings are valid"

5. NO FAKE CITATIONS:
   Never write "(Burns, 2008)" or academic references. If you reference wisdom from the books,
   say it naturally: "There's an idea in CBT that..." or "The Stoics used to say..."

6. PRACTICAL QUESTIONS GET PRACTICAL ANSWERS:
   Headache? Quick tips, then move on. Don't turn everything into a deep discussion.
   Sleep issues? Specific suggestions. Bad habits? Concrete strategies.

7. DON'T ASSUME CRISIS:
   A headache is a headache. Not every question needs philosophical wisdom.
   Read what they're ACTUALLY asking for.

8. LENGTH MATTERS:
   - Greetings/casual: 1-2 sentences
   - Simple questions: 2-4 sentences  
   - Deeper issues: Can go longer, but stay focused
   - Never ramble or repeat yourself

9. WARMTH WITHOUT BEING CLINICAL:
   Sound like a caring friend, not a therapist or wellness coach.
   "I'm here with you" not "I'm here to support your journey"

10. CONTEXT AWARENESS:
    If they say "no" or "it's not working", acknowledge and pivot:
    "Okay, let's try something else..." not "Let me explain why that approach..."

Remember: You're a companion, not a clinical tool. Be real, be warm, be present."""

def build_prompt(question: str, retrieved: list, history: list, active_source: str | None = None) -> str:
    """Build prompt with conversation history for continuity"""
    
    # Format conversation history
    history_text = ""
    if history:
        history_text = "RECENT CONVERSATION:\n"
        for turn in history[-6:]:  # Last 6 turns (3 exchanges)
            role = turn["role"]
            content = turn["content"][:200]  # Truncate for context window
            history_text += f"{role}: {content}\n"
        history_text += "\n"
    
    # Only use highly relevant passages, keep brief
    blocks = []
    for r in retrieved:
        if r["text"] and r.get("score", 0) > MIN_CONTEXT_SCORE:
            blocks.append(f"[{r['citation']}]: {r['text'][:300]}")
        if len(blocks) >= 3:
            break

    context_section = ""
    if blocks:
        context_section = (
            "WISDOM FROM YOUR KNOWLEDGE BASE (only use if directly relevant):\n"
            + "\n".join(blocks) + "\n\n"
        )

    active_source_rule = ""
    if active_source:
        active_source_rule = (
            f"- Active source for this turn: {source_display_name(active_source)}. "
            "Use only this source for the answer and do not bring in ideas or citations from other books unless the user explicitly asks to compare sources.\n"
        )

    return (
        f"{PERSONA}\n\n"
        "SOURCE GROUNDING RULES:\n"
        f"{active_source_rule}"
        "- If the user asks for a verse, chapter, quote, or wording from a specific text, answer only from that exact source.\n"
        "- If the user asks how to apply, understand, or live a Bhagavad Gita verse from the recent conversation, stay anchored to that verse.\n"
        "- Do not mix books when the user names a specific book or follows up on a specific book.\n"
        "- If the user explicitly asks to compare multiple sources, then and only then use multiple sources.\n"
        "- If the provided context does not contain the requested passage, say you could not find it instead of guessing.\n\n"
        f"{history_text}"
        f"{context_section}"
        f"User: {question}\n\n"
        f"KrishnaAI (respond naturally, warmly, and appropriately to what they just said):"
    )

# ── Ollama caller with better parameters ──────────────────────────────────────
def call_ollama(prompt, model=OLLAMA_MODEL, url=OLLAMA_URL, timeout=OLLAMA_TIMEOUT):
    payload = {
        "model": model, 
        "prompt": prompt, 
        "stream": False,
        "options": {
            "temperature": 0.35,      # Slightly higher for more natural responses
            "num_predict": 300,       # Allow slightly longer responses
            "repeat_penalty": 1.35,   # Prevent repetition
            "top_p": 0.9,             # Nucleus sampling for variety
            "top_k": 40               # Consider top 40 tokens
        }
    }
    try:
        resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=timeout)
        data = resp.json()
    except Exception as e:
        raise RuntimeError(f"Ollama request failed: {e}")

    def extract(d):
        if not isinstance(d, dict): return None
        for k in ("response", "text", "output", "result"):
            v = d.get(k)
            if isinstance(v, str) and v.strip(): return v.strip()
        for arr_k in ("choices", "outputs"):
            arr = d.get(arr_k)
            if isinstance(arr, list) and arr:
                parts = []
                for item in arr:
                    t = (item.get("text") or
                         (item.get("message") or {}).get("content") or
                         item.get("response"))
                    if isinstance(t, str) and t.strip(): parts.append(t.strip())
                if parts: return "\n".join(parts)
        return None

    text = extract(data)
    if not text:
        return "I'm having trouble reaching my knowledge engine right now. Could you try asking again?"
    
    # Clean up any artifacts
    text = text.strip()
    # Remove "KrishnaAI:" if the model echoed it
    if text.lower().startswith("krishnaai:"):
        text = text[10:].strip()
    
    return text

# ── Crisis check ──────────────────────────────────────────────────────────────
def check_crisis(question: str) -> str | None:
    if CRISIS_PATTERNS.search(question):
        return CRISIS_RESPONSE
    return None

def is_high_risk_crisis(question: str) -> bool:
    return bool(question and HIGH_RISK_PATTERNS.search(question))

# ── Conversation history management ───────────────────────────────────────────
class ConversationMemory:
    def __init__(self, max_turns=MAX_HISTORY):
        self.history = deque(maxlen=max_turns * 2)  # user + ai = 2 messages per turn
        
    def add_user(self, message: str):
        self.history.append({"role": "User", "content": message, "timestamp": datetime.now()})
    
    def add_ai(self, message: str):
        self.history.append({"role": "KrishnaAI", "content": message, "timestamp": datetime.now()})
    
    def get_history(self):
        return list(self.history)
    
    def clear(self):
        self.history.clear()

# ── Terminal loop with memory ─────────────────────────────────────────────────
def main():
    index, chunks, meta, embed = load_system()
    memory = ConversationMemory()
    
    print("\n" + "="*60)
    print("🪷  KrishnaAI Mental Wellness Companion")
    print("="*60)
    print("I'm here to listen and chat with you about whatever's on your mind.")
    print("Type 'clear' to start a fresh conversation, or 'exit' to quit.\n")
    
    while True:
        q = input("You: ").strip()
        if not q: continue
        
        if q.lower() in ("exit", "quit"):
            print("\n🙏 Take care of yourself. I'm here whenever you need me.\n")
            break
            
        if q.lower() == "clear":
            memory.clear()
            print("\n✨ Starting fresh. What's on your mind?\n")
            continue

        # Check for crisis
        crisis = check_crisis(q)
        if crisis:
            print(f"\nKrishnaAI:\n{crisis}\n\n{'─'*60}\n")
            memory.add_user(q)
            memory.add_ai(crisis)
            continue

        # Add to memory
        memory.add_user(q)

        try:
            exact_match = answer_exact_reference(q)
            if exact_match:
                answer = exact_match["answer"]
                print(f"\nKrishnaAI:\n{answer}\n")
                print(f"📚 Drew from: {exact_match['citation']}")
                print(f"\n{'─'*60}\n")
                memory.add_ai(answer)
                continue

            # Retrieve relevant passages
            q_emb     = embed_query(embed, q)
            retrieved = retrieve(index, chunks, meta, q_emb, question=q)
            
            # Build prompt with history
            prompt = build_prompt(q, retrieved, memory.get_history())
            
            # Get response
            print("\n💭 ", end="", flush=True)
            t = time.time()
            answer = call_ollama(prompt)
            elapsed = time.time() - t
            
            # Display response
            print(f"\nKrishnaAI:\n{answer}\n")
            
            # Show sources if relevant (only for non-casual questions)
            if len(q.split()) > 3 and not any(greeting in q.lower() for greeting in ["hi", "hello", "hey", "sup"]):
                sources = [r['citation'] for r in retrieved if r.get('score', 0) > MIN_CONTEXT_SCORE]
                if sources:
                    print(f"📚 Drew from: {', '.join(set(sources[:3]))}")
            
            print(f"⏱️  {elapsed:.1f}s\n{'─'*60}\n")
            
            # Add to memory
            memory.add_ai(answer)
            
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            continue

if __name__ == "__main__":
    main()
