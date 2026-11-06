/* KrishnaAI Enhanced Frontend — Natural Conversations with Memory */

const API_BASE = window.location.protocol === "file:" ? "http://localhost:5000" : "";
const API = `${API_BASE}/api/ask`;
const CLEAR_API = `${API_BASE}/api/clear`;
const FEEDBACK_API = `${API_BASE}/api/feedback`;
const NEAREST_HELP_API = `${API_BASE}/api/nearest_help`;

const $messages = document.getElementById("messages");
const $welcome  = document.getElementById("welcome");
const $input    = document.getElementById("input");
const $sendBtn  = document.getElementById("send-btn");
const $sidebar  = document.getElementById("sidebar");

let chatStarted = false;
let isTyping = false;

// ── Auto-resize textarea ──────────────────────────────────────────────
$input.addEventListener("input", () => {
  $input.style.height = "auto";
  $input.style.height = Math.min($input.scrollHeight, 200) + "px";
});

$input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !isTyping) {
    e.preventDefault();
    sendMessage();
  }
});

// Auto-focus input on load
window.addEventListener("load", () => {
  setTimeout(() => $input.focus(), 100);
});

// ── Sidebar toggle ────────────────────────────────────────────────────
function toggleSidebar() {
  $sidebar.classList.toggle("collapsed");
}

// ── New chat ──────────────────────────────────────────────────────────
async function newChat() {
  // Clear server-side history
  try {
    await fetch(CLEAR_API, {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json" }
    });
  } catch (err) {
    console.error("Failed to clear history:", err);
  }

  // Clear UI
  $messages.innerHTML = "";
  chatStarted = false;
  $welcome.classList.remove("hidden");
  $input.value = "";
  $input.style.height = "auto";
  $input.focus();
}

// ── Send prompt from chips ────────────────────────────────────────────
function sendPrompt(btn) {
  if (chatStarted || isTyping || $welcome.classList.contains("hidden")) return;
  $input.value = btn.textContent;
  sendMessage();
}

// ── Enhanced markdown rendering ───────────────────────────────────────
function renderText(raw) {
  let html = raw
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    // bold: **text** or __text__
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/__(.*?)__/g, "<strong>$1</strong>")
    // italic: *text* or _text_
    .replace(/\*(.*?)\*/g, "<em>$1</em>")
    .replace(/_(.*?)_/g, "<em>$1</em>")
    // bullet lines: lines starting with • or - or *
    .replace(/^[•\-\*]\s+(.+)/gm, "<li>$1</li>")
    // numbered lists: 1. item
    .replace(/^\d+\.\s+(.+)/gm, "<li>$1</li>")
    // line breaks → <br>
    .replace(/\n/g, "<br>");

  // Wrap consecutive <li> in <ul>
  html = html.replace(/(<li>.*?<\/li>(<br>)?)+/gs, (m) => {
    const items = m.replace(/<br>/g, "");
    return "<ul style='margin:8px 0 8px 20px; line-height:1.6;'>" + items + "</ul>";
  });

  return html;
}

// ── Add a message bubble ──────────────────────────────────────────────
function addMessage(role, text, citations = [], isCrisis = false, feedback = null) {
  if (!chatStarted) {
    chatStarted = true;
    $welcome.classList.add("hidden");
  }

  const div = document.createElement("div");
  div.className = "msg " + role + (isCrisis ? " crisis" : "");

  const avatarContent = role === "user"
    ? "You"
    : `<svg viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg" style="width:20px;height:20px">
        <ellipse cx="24" cy="14" rx="4" ry="9" fill="currentColor" opacity="0.8"/>
        <ellipse cx="24" cy="14" rx="4" ry="9" fill="currentColor" opacity="0.8" transform="rotate(60 24 24)"/>
        <ellipse cx="24" cy="14" rx="4" ry="9" fill="currentColor" opacity="0.8" transform="rotate(120 24 24)"/>
        <ellipse cx="24" cy="14" rx="4" ry="9" fill="currentColor" opacity="0.8" transform="rotate(180 24 24)"/>
        <ellipse cx="24" cy="14" rx="4" ry="9" fill="currentColor" opacity="0.8" transform="rotate(240 24 24)"/>
        <ellipse cx="24" cy="14" rx="4" ry="9" fill="currentColor" opacity="0.8" transform="rotate(300 24 24)"/>
        <circle cx="24" cy="24" r="5" fill="currentColor"/>
       </svg>`;

  const name = role === "user" ? "You" : "KrishnaAI";

  const citationsHtml = (citations && citations.length > 0)
    ? `<div class="msg-citations">${citations.map(c => `<span class="citation-tag">${c}</span>`).join("")}</div>`
    : "";

  div.innerHTML = `
    <div class="msg-avatar">${avatarContent}</div>
    <div class="msg-body">
      <div class="msg-name">${name}</div>
      <div class="msg-text">${role === "ai" ? renderText(text) : escapeHtml(text)}</div>
      ${role === "ai" ? citationsHtml : ""}
    </div>`;

  if (role === "ai" && feedback && feedback.responseId) {
    const actions = document.createElement("div");
    actions.className = "feedback-actions";
    actions.innerHTML = `
      <button type="button" class="feedback-btn" data-rating="like" title="Mark this answer as helpful">Like</button>
      <button type="button" class="feedback-btn" data-rating="dislike" title="Mark this answer as not helpful">Dislike</button>
      <span class="feedback-status" aria-live="polite"></span>`;

    actions.querySelectorAll(".feedback-btn").forEach((btn) => {
      btn.addEventListener("click", () => submitFeedback(btn, feedback));
    });

    div.querySelector(".msg-body").appendChild(actions);
  }

  $messages.appendChild(div);
  
  // Smooth scroll with animation
  setTimeout(() => {
    div.scrollIntoView({ behavior: "smooth", block: "end" });
  }, 50);
  
  return div;
}

function escapeHtml(t) {
  return t.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/\n/g,"<br>");
}

async function submitFeedback(btn, feedback) {
  const actions = btn.closest(".feedback-actions");
  const status = actions.querySelector(".feedback-status");
  const rating = btn.dataset.rating;

  actions.querySelectorAll(".feedback-btn").forEach((item) => {
    item.disabled = true;
    item.classList.toggle("selected", item === btn);
  });
  status.textContent = "Saving...";

  try {
    const res = await fetch(FEEDBACK_API, {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        response_id: feedback.responseId,
        rating,
        question: feedback.question,
        answer: feedback.answer,
        citations: feedback.citations,
        active_source: feedback.activeSource
      })
    });

    if (!res.ok) throw new Error("Feedback save failed");
    status.textContent = rating === "like" ? "Saved as helpful" : "Saved for improvement";
  } catch (err) {
    actions.querySelectorAll(".feedback-btn").forEach((item) => {
      item.disabled = false;
      item.classList.remove("selected");
    });
    status.textContent = "Could not save";
    console.error("Feedback error:", err);
  }
}

// ── Enhanced typing indicator ─────────────────────────────────────────
function addTyping() {
  const div = document.createElement("div");
  div.className = "msg ai typing-msg";
  div.id = "typing-indicator";
  div.innerHTML = `
    <div class="msg-avatar">
      <svg viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg" style="width:20px;height:20px">
        <ellipse cx="24" cy="14" rx="4" ry="9" fill="currentColor" opacity="0.8"/>
        <ellipse cx="24" cy="14" rx="4" ry="9" fill="currentColor" opacity="0.8" transform="rotate(60 24 24)"/>
        <ellipse cx="24" cy="14" rx="4" ry="9" fill="currentColor" opacity="0.8" transform="rotate(120 24 24)"/>
        <ellipse cx="24" cy="14" rx="4" ry="9" fill="currentColor" opacity="0.8" transform="rotate(180 24 24)"/>
        <ellipse cx="24" cy="14" rx="4" ry="9" fill="currentColor" opacity="0.8" transform="rotate(240 24 24)"/>
        <ellipse cx="24" cy="14" rx="4" ry="9" fill="currentColor" opacity="0.8" transform="rotate(300 24 24)"/>
        <circle cx="24" cy="24" r="5" fill="currentColor"/>
      </svg>
    </div>
    <div class="msg-body">
      <div class="msg-name">KrishnaAI</div>
      <div class="msg-text"><div class="typing-dots"><span></span><span></span><span></span></div></div>
    </div>`;
  $messages.appendChild(div);
  div.scrollIntoView({ behavior: "smooth", block: "end" });
}

function removeTyping() {
  const el = document.getElementById("typing-indicator");
  if (el) el.remove();
}

function formatPlace(label, place) {
  if (!place) {
    return `${label}: I could not verify a nearby ${label.toLowerCase()} from the available map data.`;
  }

  const lines = [
    `${label}: ${place.name}`,
    place.address ? `Address: ${place.address}` : "",
    Number.isFinite(place.distance_km) ? `Distance: ${place.distance_km} km` : "",
    place.phone ? `Phone: ${place.phone}` : "",
    place.website ? `Website: ${place.website}` : "",
    place.lat && place.lon ? `Map: https://www.google.com/maps?q=${place.lat},${place.lon}` : "",
    place.source ? `Source: ${place.source}` : ""
  ].filter(Boolean);

  return lines.join("\n");
}

async function askForCrisisLocation() {
  if (!navigator.geolocation) {
    addMessage(
      "ai",
      "I cannot access location from this browser. If you might act on these thoughts, please call emergency services now or go to the nearest emergency room. If you are in India, you can also call iCall at 9152987821 or Vandrevala Foundation at 1860-2662-345.",
      [],
      true
    );
    return;
  }

  addMessage(
    "ai",
    "I am worried about your immediate safety. If you allow location access, I will try to find real nearby hospital and doctor/clinic details from map data. If you are in immediate danger, call emergency services now.",
    [],
    true
  );

  navigator.geolocation.getCurrentPosition(async (pos) => {
    try {
      const res = await fetch(NEAREST_HELP_API, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          lat: pos.coords.latitude,
          lon: pos.coords.longitude
        })
      });
      const data = await res.json();

      if (!res.ok || data.error) {
        addMessage(
          "ai",
          "I could not verify nearby hospital or doctor details right now. Please call emergency services now, or ask someone near you to take you to the closest emergency room. If you are in India, iCall is 9152987821 and Vandrevala Foundation is 1860-2662-345.",
          [],
          true
        );
        return;
      }

      addMessage(
        "ai",
        [
          data.message,
          "",
          formatPlace("Hospital", data.hospital),
          "",
          formatPlace("Doctor/clinic", data.doctor)
        ].join("\n"),
        [],
        true
      );
    } catch (e) {
      addMessage(
        "ai",
        "I could not verify nearby places right now. Please call emergency services now, or go to the nearest emergency room with someone you trust.",
        [],
        true
      );
    }
  }, () => {
    addMessage(
      "ai",
      "Location access was not shared. That is okay. Please call emergency services now if you might hurt yourself, or ask a nearby person to stay with you and help you get to the closest emergency room.",
      [],
      true
    );
  }, {
    enableHighAccuracy: true,
    timeout: 10000,
    maximumAge: 60000
  });
}

// ── Main send with better error handling ──────────────────────────────
async function sendMessage() {
  const text = $input.value.trim();
  if (!text || isTyping) return;

  // Show user message
  addMessage("user", text);
  $input.value = "";
  $input.style.height = "auto";
  $sendBtn.disabled = true;
  isTyping = true;

  // Typing indicator
  addTyping();

  try {
    const res = await fetch(API, {
      method: "POST",
      credentials: "include",  // Important for session cookies
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
    });

    const data = await res.json();
    removeTyping();

    if (data.error) {
      addMessage("ai", 
        "I'm having a bit of trouble right now. Could you try asking again?", 
        [], false);
      return;
    }

    const isCrisis = !!data.crisis;
    
    // Show citations only for non-greetings and substantial responses
    const showCitations = data.citations && data.citations.length > 0 && text.split(" ").length > 3;
    const citations = showCitations ? data.citations : [];
    
    addMessage("ai", data.answer, citations, isCrisis, {
      responseId: data.response_id,
      question: text,
      answer: data.answer,
      citations,
      activeSource: data.active_source || null
    });
    if (data.ask_location) {
      await askForCrisisLocation();
    }

  } catch (err) {
    removeTyping();
    addMessage("ai",
      "I'm having trouble connecting right now. Please make sure the server is running and try again.",
      [], false
    );
    console.error("API error:", err);
  } finally {
    $sendBtn.disabled = false;
    isTyping = false;
    $input.focus();
  }
}

// ── Greeting detection for better UX ──────────────────────────────────
function isGreeting(text) {
  const greetings = ["hi", "hello", "hey", "sup", "yo", "hiya", "howdy"];
  const lowerText = text.toLowerCase().trim();
  return greetings.some(g => lowerText === g || lowerText.startsWith(g + " "));
}

// ── Add helpful tooltips ──────────────────────────────────────────────
window.addEventListener("load", () => {
  // Add tooltip to New Conversation button
  const newChatBtn = document.querySelector(".nav-item");
  if (newChatBtn) {
    newChatBtn.title = "Start a fresh conversation (clears memory)";
  }

});
