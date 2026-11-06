# server_enhanced.py — KrishnaAI with Session Memory
import json, math, os, traceback, uuid
import requests
from flask import Flask, request, jsonify, send_from_directory, session
from flask_cors import CORS
from collections import defaultdict
from datetime import datetime, timedelta

rt = None
app = Flask(__name__, static_folder=None)
app.secret_key = os.urandom(24)  # For session management
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)
CORS(app, supports_credentials=True)

POSSIBLE_FRONTENDS = [
    os.environ.get('FRONTEND_DIR'),
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend')),
    os.path.abspath(os.path.join(os.path.dirname(__file__), 'frontend')),
]
FRONTEND_DIR = next((p for p in POSSIBLE_FRONTENDS if p and os.path.isdir(p)), None)

INDEX = CHUNKS = META = EMBED = None

# In-memory conversation storage (user_id -> list of messages)
# In production, use Redis or a database
conversations = defaultdict(list)
MAX_HISTORY_PER_USER = 12  # 6 exchanges
PLACES_TIMEOUT = 8
PLACES_RADIUS_METERS = 5000
FEEDBACK_DIR = os.path.join(os.path.dirname(__file__), "feedback")
FEEDBACK_FILE = os.path.join(FEEDBACK_DIR, "feedback.jsonl")

def load_rag():
    global INDEX, CHUNKS, META, EMBED, rt
    try:
        if rt is None:
            import importlib.util
            path = os.path.join(os.path.dirname(__file__), 'rag_terminal.py')
            spec = importlib.util.spec_from_file_location('rag_terminal', path)
            _rt  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(_rt)
            rt = _rt
        INDEX, CHUNKS, META, EMBED = rt.load_system()
    except Exception as e:
        print("Warning: RAG load failed:", e)
        traceback.print_exc()

def get_user_id():
    """Get or create user session ID"""
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
        session.permanent = True
    return session['user_id']

def get_conversation_history(user_id):
    """Get conversation history for user"""
    return conversations[user_id]

def add_to_history(user_id, role, content):
    """Add message to conversation history"""
    conversations[user_id].append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    })
    # Keep only recent messages
    if len(conversations[user_id]) > MAX_HISTORY_PER_USER:
        conversations[user_id] = conversations[user_id][-MAX_HISTORY_PER_USER:]

def append_feedback(record):
    os.makedirs(FEEDBACK_DIR, exist_ok=True)
    with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def km_between(lat1, lon1, lat2, lon2):
    radius_km = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return radius_km * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def format_osm_address(tags):
    parts = [
        tags.get("addr:housenumber"),
        tags.get("addr:street"),
        tags.get("addr:suburb") or tags.get("addr:neighbourhood"),
        tags.get("addr:city") or tags.get("addr:town") or tags.get("addr:village"),
        tags.get("addr:postcode"),
    ]
    return ", ".join(p for p in parts if p)

def place_from_osm_element(element, user_lat, user_lon):
    tags = element.get("tags") or {}
    lat = element.get("lat") or (element.get("center") or {}).get("lat")
    lon = element.get("lon") or (element.get("center") or {}).get("lon")
    if lat is None or lon is None:
        return None

    name = tags.get("name")
    if not name:
        return None

    return {
        "name": name,
        "address": format_osm_address(tags),
        "phone": tags.get("phone") or tags.get("contact:phone"),
        "website": tags.get("website") or tags.get("contact:website"),
        "lat": lat,
        "lon": lon,
        "distance_km": round(km_between(user_lat, user_lon, lat, lon), 2),
        "source": "OpenStreetMap",
    }

def query_overpass(lat, lon, place_kind):
    if place_kind == "hospital":
        clauses = [
            'node(around:{radius},{lat},{lon})["amenity"="hospital"];',
            'way(around:{radius},{lat},{lon})["amenity"="hospital"];',
            'relation(around:{radius},{lat},{lon})["amenity"="hospital"];',
        ]
    else:
        clauses = [
            'node(around:{radius},{lat},{lon})["amenity"="doctors"];',
            'way(around:{radius},{lat},{lon})["amenity"="doctors"];',
            'relation(around:{radius},{lat},{lon})["amenity"="doctors"];',
            'node(around:{radius},{lat},{lon})["healthcare"~"doctor|psychiatrist|psychotherapist"];',
            'way(around:{radius},{lat},{lon})["healthcare"~"doctor|psychiatrist|psychotherapist"];',
            'relation(around:{radius},{lat},{lon})["healthcare"~"doctor|psychiatrist|psychotherapist"];',
        ]

    body = "\n      ".join(
        clause.format(radius=PLACES_RADIUS_METERS, lat=lat, lon=lon)
        for clause in clauses
    )
    query = f"""
    [out:json][timeout:10];
    (
      {body}
    );
    out center tags 20;
    """
    response = requests.post(
        "https://overpass-api.de/api/interpreter",
        data={"data": query},
        timeout=PLACES_TIMEOUT,
        headers={"User-Agent": "KrishnaAI/1.0"},
    )
    response.raise_for_status()
    places = []
    for element in response.json().get("elements", []):
        place = place_from_osm_element(element, lat, lon)
        if place:
            places.append(place)
    places.sort(key=lambda item: item["distance_km"])
    return places

@app.route("/api/ask", methods=["POST"])
def ask():
    data     = request.json or {}
    question = data.get("question") or data.get("message") or ""
    if not question.strip():
        return jsonify({"error": "Empty question"}), 400

    # Get user session
    user_id = get_user_id()
    history = get_conversation_history(user_id)
    response_id = str(uuid.uuid4())

    # 1. Crisis gate — always first
    if rt:
        crisis = rt.check_crisis(question)
        if crisis:
            high_risk = rt.is_high_risk_crisis(question)
            add_to_history(user_id, "User", question)
            add_to_history(user_id, "KrishnaAI", crisis)
            return jsonify({
                "response_id": response_id,
                "answer": crisis,
                "crisis": True,
                "risk": "high" if high_risk else "crisis",
                "ask_location": high_risk,
                "citations": []
            })

    # 2. RAG not ready
    if INDEX is None:
        return jsonify({
            "response_id": response_id,
            "answer": "My knowledge system is still loading. Please try again shortly.",
            "warning": "RAG not loaded"
        })

    # 3. Add user message to history
    add_to_history(user_id, "User", question)

    exact_match = rt.answer_exact_reference(question) if rt else None
    if exact_match:
        answer = exact_match["answer"]
        add_to_history(user_id, "KrishnaAI", answer)
        return jsonify({
            "response_id": response_id,
            "answer": answer,
            "citations": [exact_match["citation"]],
            "retrieved": exact_match["retrieved"],
            "conversation_length": len(conversations[user_id]),
            "grounded_lookup": True,
        })

    # 4. Normal RAG flow with history context
    active_source = rt.resolve_active_source(question, history) if rt else None
    contextual_gita_ref = rt.resolve_contextual_gita_reference(question, history) if rt else None
    if contextual_gita_ref:
        retrieved = rt.gita_reference_retrieval(*contextual_gita_ref)
        active_source = "bhagavad gita"
    else:
        try:
            q_emb     = rt.embed_query(EMBED, question)
            retrieved = rt.retrieve(INDEX, CHUNKS, META, q_emb, question=question, source_key=active_source)
        except Exception as e:
            return jsonify({"error": f"Retrieval error: {e}"}), 500

    # Build prompt with conversation history
    prompt = rt.build_prompt(question, retrieved, history, active_source=active_source)

    try:
        answer = rt.call_ollama(prompt)
    except Exception as e:
        return jsonify({"error": f"LLM error: {e}"}), 500

    # Add AI response to history
    add_to_history(user_id, "KrishnaAI", answer)

    # Prepare response
    min_score = getattr(rt, "MIN_CONTEXT_SCORE", 0.32)
    citations = list({r["citation"] for r in retrieved if r.get("citation") and r.get("score", 0) > min_score})
    trimmed   = [{"score": r["score"], "text": r["text"][:1500], "citation": r["citation"]} 
                 for r in retrieved if r.get("score", 0) > min_score][:3]  # Only top 3 relevant

    return jsonify({
        "response_id": response_id,
        "answer": answer, 
        "citations": citations, 
        "retrieved": trimmed,
        "active_source": active_source,
        "conversation_length": len(conversations[user_id])
    })

@app.route("/api/feedback", methods=["POST"])
def save_feedback():
    data = request.json or {}
    rating = data.get("rating")
    if rating not in {"like", "dislike"}:
        return jsonify({"error": "rating must be 'like' or 'dislike'"}), 400

    record = {
        "feedback_id": str(uuid.uuid4()),
        "response_id": data.get("response_id") or str(uuid.uuid4()),
        "rating": rating,
        "question": data.get("question", ""),
        "answer": data.get("answer", ""),
        "citations": data.get("citations", []),
        "active_source": data.get("active_source"),
        "user_id": get_user_id(),
        "created_at": datetime.now().isoformat(),
    }
    append_feedback(record)
    return jsonify({"status": "saved", "feedback_id": record["feedback_id"]})

@app.route("/api/clear", methods=["POST"])
def clear_history():
    """Clear conversation history for current user"""
    user_id = get_user_id()
    conversations[user_id] = []
    return jsonify({"status": "cleared", "message": "Conversation history cleared"})

@app.route("/api/history", methods=["GET"])
def get_history():
    """Get conversation history for current user"""
    user_id = get_user_id()
    history = get_conversation_history(user_id)
    return jsonify({"history": history, "length": len(history)})

@app.route("/health")
def health():
    return jsonify({
        "status": "ok", 
        "rag_loaded": INDEX is not None,
        "index_count": getattr(INDEX, 'ntotal', None),
        "active_conversations": len(conversations)
    })

@app.route("/admin/reload", methods=["POST"])
def reload_rag():
    global INDEX, CHUNKS, META, EMBED, rt
    rt = None
    try:
        load_rag()
        return jsonify({"status": "loaded", "index_count": getattr(INDEX, 'ntotal', None)})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route("/", defaults={"path": "index.html"})
@app.route("/<path:path>")
def serve_frontend(path):
    if FRONTEND_DIR and os.path.isfile(os.path.join(FRONTEND_DIR, path)):
        return send_from_directory(FRONTEND_DIR, path)
    if FRONTEND_DIR and os.path.isfile(os.path.join(FRONTEND_DIR, "index.html")):
        return send_from_directory(FRONTEND_DIR, "index.html")
    return jsonify({"error": "Frontend not found. Set FRONTEND_DIR env var."}), 404

# ── Location-based nearest help endpoint ─────────────────────────────
@app.route("/api/nearest_help", methods=["POST"])
def nearest_help():
    data = request.json or {}
    try:
        lat = float(data.get("lat"))
        lon = float(data.get("lon"))
    except (TypeError, ValueError):
        return jsonify({"error": "Missing coordinates."}), 400

    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return jsonify({"error": "Invalid coordinates."}), 400

    try:
        hospitals = query_overpass(lat, lon, "hospital")
        doctors = query_overpass(lat, lon, "doctor")
    except requests.RequestException as e:
        return jsonify({
            "error": "Could not verify nearby places right now.",
            "detail": str(e),
            "source": "OpenStreetMap"
        }), 502

    return jsonify({
        "hospital": hospitals[0] if hospitals else None,
        "doctor": doctors[0] if doctors else None,
        "source": "OpenStreetMap",
        "radius_meters": PLACES_RADIUS_METERS,
        "message": (
            "These are the nearest verified OpenStreetMap results I could find. "
            "If you are in immediate danger, call local emergency services now."
        )
    })

if __name__ == "__main__":
    print("Starting KrishnaAI Enhanced Server")
    print(f"Frontend: {FRONTEND_DIR or '<not found>'}")
    print("Memory: Session-based (24h retention)")
    load_rag()
    print("Server running at http://localhost:5000")
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)
