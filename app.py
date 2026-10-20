import io
import json
import re
from typing import List, Tuple, Optional
import cv2
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
import streamlit as st

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Make Your Notes Smart",
    layout="wide",
    page_icon="🎓",
    initial_sidebar_state="expanded",
)

try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except ImportError:
    CANVAS_AVAILABLE = False


# --- 2. Clean Modern Styling ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 10% 10%, #0f172a 0%, #090d16 100%) !important;
        color: #f1f5f9;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #0b0f19 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.07);
    }
    
    .glass-panel {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        backdrop-filter: blur(14px);
    }
    
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #d946ef 100%) !important;
        border: none !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        border-radius: 12px !important;
        padding: 0.65rem 1.4rem !important;
        box-shadow: 0 4px 18px rgba(99, 102, 241, 0.35) !important;
    }
    
    .stTextInput input, .stSelectbox [data-baseweb="select"] {
        border-radius: 10px !important;
        background-color: #131b2e !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        color: #f8fafc !important;
    }
</style>
""", unsafe_allow_html=True)


# --- 3. Secure API Key & Logic ---
def get_system_api_key() -> str:
    try:
        if "GROQ_API_KEY" in st.secrets and st.secrets["GROQ_API_KEY"].strip():
            return st.secrets["GROQ_API_KEY"].strip()
    except Exception:
        pass
    return ""


def clean_document_mask(img_bgr: np.ndarray, mask: np.ndarray, radius: int = 4) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated_mask = cv2.dilate(mask, kernel, iterations=2)
    return cv2.inpaint(img_bgr, dilated_mask, radius, cv2.INPAINT_TELEA)


def is_academic_noise(text: str) -> bool:
    low = text.lower().strip()
    if re.match(r"^\d{1,3}[\.\)]\s+(what|why|how|give|explain|state|which|compare|calculate)\b", low):
        return True
    if len(re.findall(r"\b\d{2,4}\b", text)) >= 4:
        return True
    noisy_patterns = [
        "recommended reading", "practice set", "review questions",
        "key terms", "chapter summary", "exercises", "solutions manual"
    ]
    if any(p in low for p in noisy_patterns) and len(text.split()) < 60:
        return True
    return len(text.split()) < 15


def extract_meaningful_pdf_blocks(pdf_bytes: bytes, start_page: int, end_page: int) -> List[Tuple[int, str]]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(doc)
    start_idx = max(0, start_page - 1)
    end_idx = min(end_page, total_pages)

    valid_blocks = []
    for page_idx in range(start_idx, end_idx):
        page = doc[page_idx]
        blocks = page.get_text("blocks")
        for b in blocks:
            clean = re.sub(r"(\w+)-\n(\w+)", r"\1\2", b[4])
            clean = re.sub(r"\s+", " ", clean).strip()
            if not is_academic_noise(clean):
                valid_blocks.append((page_idx + 1, clean))
    return valid_blocks


def rank_and_retrieve_context(blocks: List[Tuple[int, str]], topic: str, top_k: int = 15) -> str:
    topic_clean = topic.strip().lower()
    topic_tokens = [t for t in re.findall(r"\w+", topic_clean) if len(t) > 2]

    if not topic_tokens:
        return "\n\n".join([f"[Page {p}] {t}" for p, t in blocks[:top_k]])

    scored = []
    for page_num, text in blocks:
        text_lower = text.lower()
        word_count = len(text.split())
        phrase_weight = 15 if topic_clean in text_lower else 0
        token_hits = sum(text_lower.count(t) for t in topic_tokens)
        score = (phrase_weight + token_hits) / (word_count ** 0.35)
        if score > 0:
            scored.append((score, page_num, text))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_matches = scored[:top_k] if scored else blocks[:top_k]
    return "\n\n".join([f"[Page {p}]\n{t}" for _, p, t in top_matches])


def create_pdf_from_text(title: str, text: str) -> bytes:
    """Notes ko clean printable PDF bytes me convert karta hai."""
    doc = fitz.open()
    margin = 40
    width, height = 595, 842  # A4 Size
    
    clean_lines = []
    for raw_l in text.split("\n"):
        clean_l = raw_l.replace("**", "").replace("### ", "").replace("## ", "").replace("# ", "")
        clean_lines.append(clean_l)
        
    current_page = doc.new_page(width=width, height=height)
    y = margin
    
    # Title Header
    current_page.insert_text((margin, y), f"{title.upper()} - STUDY NOTES", fontsize=14, fontname="helv", color=(0.2, 0.2, 0.8))
    y += 25
    
    for line in clean_lines:
        if y > height - margin:
            current_page = doc.new_page(width=width, height=height)
            y = margin
        
        # Word wrap approx 90 chars
        chunks = [line[i:i+90] for i in range(0, len(line), 90)] if line else [""]
        for c in chunks:
            if y > height - margin:
                current_page = doc.new_page(width=width, height=height)
                y = margin
            current_page.insert_text((margin, y), c, fontsize=10, fontname="helv", color=(0.1, 0.1, 0.1))
            y += 14
        y += 4

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def generate_student_notes(context: str, topic: str, depth_level: str, api_key: str) -> str:
    if not api_key:
        return(
            "⚠️ **Groq API Key Required!** \n"
            "1. **[console.groq.com/keys](https://console.groq.com/keys)** make free key .\n"
            "Please provide free Groq API key in the sidebar."
        )

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key.strip(), base_url="https://api.groq.com/openai/v1")

        system_instruction = (
            "You are an elite Computer Science Professor and Technical Author. "
            "Write comprehensive, exhaustive academic notes covering every mode, category, "
            "and concept present in the context. Never leave sections half-written."
        )

        user_prompt = f"""
TARGET TOPIC: {topic}
REVISION DEPTH: {depth_level}

TEXTBOOK CONTEXT:
\"\"\"
{context[:16000]}
\"\"\"

Create complete, comprehensive, and student-friendly notes with this exact structure:

# 📚 {topic.title()} — Complete Comprehensive Revision Notes

## 1. Concept Overview & Core Definition
- **Definition:** Clear technical definition.
- **Architectural Purpose:** Why is this concept essential?
- **Real-World Intuition:** Everyday analogy.

## 2. Exhaustive Classification & All Sub-Parts
- Detail EVERY mode/type present in the context.
- For each sub-part explain:
  * **Working Principle & Data Flow**
  * **Key Operational Characteristics**
  * **Real-world Example**

## 3. Comprehensive Comparative Analysis (Table)
Create a full Markdown comparison table across key parameters.

## 4. Engineering Trade-offs & Bottlenecks
- **Core Advantages:** (Detailed bullet points)
- **Bottlenecks & Constraints:** (Realistic drawbacks)

## 5. High-Yield Exam & Viva Questions
Provide 3 frequently asked exam/interview questions with scoring answers.
"""

        # Auto-detect live chat models directly from your Groq account
        live_chat_models = []
        try:
            m_list = client.models.list()
            for m in m_list.data:
                mid = m.id.lower()
                # Audio / Guard / Decommissioned models skip karein
                if any(x in mid for x in ["whisper", "guard", "orpheus", "tts"]):
                    continue
                live_chat_models.append(m.id)
        except Exception:
            pass

        # Priority fallback order
        candidate_pool = live_chat_models + [
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "llama-3.3-70b-versatile"
        ]

        # Unique models maintain karein
        seen = set()
        models_to_try = [m for m in candidate_pool if not (m in seen or seen.add(m))]

        last_error = None
        for model_id in models_to_try:
            try:
                res = client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=4096,
                    temperature=0.2
                )
                return res.choices[0].message.content
            except Exception as e:
                last_error = e
                continue

        return f"❌ **AI Generation Failed:** `{str(last_error)}`"

    except Exception as err:
        return f"❌ **AI Generation Failed:** `{str(err)}`"

def generate_interactive_quiz(context: str, topic: str, api_key: str):
    if not api_key:
        return None

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key.strip(), base_url="https://api.groq.com/openai/v1")

        prompt = f"""
TARGET TOPIC: {topic}
CONTEXT:
\"\"\"
{context[:14000]}
\"\"\"

Generate exactly 4 high-quality Multiple Choice Questions (MCQs) testing conceptual understanding.
Output ONLY a valid JSON array of objects.
Structure:
[
  {{
    "question": "Question text?",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "answer": "Exact matching string of correct option",
    "explanation": "Clear explanation of why this option is correct."
  }}
]
"""
        live_chat_models = []
        try:
            for m in client.models.list().data:
                mid = m.id.lower()
                if any(x in mid for x in ["whisper", "guard", "orpheus", "tts"]):
                    continue
                live_chat_models.append(m.id)
        except Exception:
            pass

        candidate_pool = live_chat_models + ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile"]
        seen = set()
        models_to_try = [m for m in candidate_pool if not (m in seen or seen.add(m))]

        for model_id in models_to_try:
            try:
                res = client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": "You are a university examiner. You ONLY output raw valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=2048,
                    temperature=0.2
                )
                raw = res.choices[0].message.content.strip()
                if raw.startswith("```"):
                    raw = re.sub(r"^```[a-zA-Z]*\n", "", raw)
                    raw = re.sub(r"```$", "", raw).strip()
                return json.loads(raw)
            except Exception:
                continue
    except Exception:
        pass
    return None


# --- 4. Hero Header ---
st.markdown("""
<div style="background: linear-gradient(180deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.4) 100%);
            border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 16px; padding: 22px 26px; margin-bottom: 22px;">
    <div style="display: flex; gap: 8px; margin-bottom: 8px;">
        <span style="background: rgba(99, 102, 241, 0.15); color: #818cf8; border: 1px solid rgba(99, 102, 241, 0.3);
                     padding: 3px 10px; border-radius: 20px; font-size: 0.72rem; font-weight: 700;">GROQ LLM</span>
        <span style="background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3);
                     padding: 3px 10px; border-radius: 20px; font-size: 0.72rem; font-weight: 700;">OPENCV TELEA</span>
        <span style="background: rgba(244, 63, 94, 0.15); color: #fb7185; border: 1px solid rgba(244, 63, 94, 0.3);
                     padding: 3px 10px; border-radius: 20px; font-size: 0.72rem; font-weight: 700;">QUIZ ENGINE</span>
    </div>
    <h2 style="margin: 0; font-size: 1.85rem; font-weight: 800; color: #f8fafc;">
        SmartNotes <span style="background: linear-gradient(135deg, #818cf8, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Studio</span>
    </h2>
    <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 0.88rem;">Synthesize revision notes, test yourself with interactive quizzes, and restore scanned textbook pages.</p>
</div>
""", unsafe_allow_html=True)

# Tabs
TAB_NOTES = "📖 AI Book Notes Synthesizer"
TAB_QUIZ = "🧠 Smart Quiz"
TAB_ERASER = "✏️ Document Eraser / Inpainter"

if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = TAB_NOTES

col_nav1, col_nav2, col_nav3 = st.columns([1.6, 1.6, 1.6])
with col_nav1:
    if st.button(TAB_NOTES, use_container_width=True, 
                 type="primary" if st.session_state["active_tab"] == TAB_NOTES else "secondary"):
        st.session_state["active_tab"] = TAB_NOTES
        st.rerun()

with col_nav2:
    if st.button(TAB_QUIZ, use_container_width=True,
                 type="primary" if st.session_state["active_tab"] == TAB_QUIZ else "secondary"):
        st.session_state["active_tab"] = TAB_QUIZ
        st.rerun()

with col_nav3:
    if st.button(TAB_ERASER, use_container_width=True,
                 type="primary" if st.session_state["active_tab"] == TAB_ERASER else "secondary"):
        st.session_state["active_tab"] = TAB_ERASER
        st.rerun()

st.markdown("---")


# --- 5. Sidebar ---
st.sidebar.title("Control Panel")
st.sidebar.caption("Dynamic tool settings")
st.sidebar.markdown("---")

if st.session_state["active_tab"] in [TAB_NOTES, TAB_QUIZ]:
    st.sidebar.subheader("📖 Chapter Settings")
    page_range_val = st.sidebar.text_input(
        "Page Range (Chapter Boundaries)",
        value="1-100",
        help="Specify the chapter start and end page to skip index pages."
    )
    if st.session_state["active_tab"] == TAB_NOTES:
        notes_depth = st.sidebar.selectbox(
            "Notes Format",
            [
                "Standard (Semester Exam Ready)",
                "Concise (Quick Revision Sheet)",
                "Comprehensive (Full Masterclass)"
            ]
        )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔑 Authentication")
    default_secret = get_system_api_key()
    user_key_input = st.sidebar.text_input(
        "Groq API Key",
        value=default_secret,
        type="password",
        placeholder="gsk_...",
        help="Paste your free API key from console.groq.com"
    )
else:
    st.sidebar.subheader("✏️ Eraser Brush Settings")
    brush_val = st.sidebar.slider("Brush Size", 5, 60, 20)
    inpaint_radius_val = st.sidebar.slider("Inpaint Radius", 1, 10, 3)


# --- 6. Main Workspace ---

# TAB 1: NOTES
if st.session_state["active_tab"] == TAB_NOTES:
    st.markdown("""
    <div class="glass-panel">
        <h4 style="margin: 0 0 6px 0; font-weight: 700; color: #f8fafc;">Synthesize Lecture & Exam Notes</h4>
        <p style="margin: 0 0 16px 0; font-size: 0.88rem; color: #94a3b8;">
            Upload your technical book or chapter PDF. The AI filters noise and structures exhaustive revision notes.
        </p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1])
    with col1:
        pdf_file = st.file_uploader("Upload Book PDF", type=["pdf"], key="pdf_input")
    with col2:
        topic_target = st.text_input("Topic to Synthesize", placeholder="e.g. Guided Media, Virtual Memory")
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Synthesize Complete Notes with AI", type="primary", use_container_width=True):
        if not pdf_file:
            st.error("Please upload your PDF files.")
        elif not topic_target.strip():
            st.error("Please provide topic's name.")
        else:
            try:
                start_p, end_p = map(int, [x.strip() for x in page_range_val.split("-")])
            except ValueError:
                start_p, end_p = 1, 100

            active_key = user_key_input.strip() or get_system_api_key()

            with st.spinner("Analyzing document, filtering noise, and generating complete notes..."):
                blocks = extract_meaningful_pdf_blocks(pdf_file.read(), start_p, end_p)
                if not blocks:
                    st.error("unable to extract PDF.")
                else:
                    context = rank_and_retrieve_context(blocks, topic_target, top_k=15)
                    notes_output = generate_student_notes(context, topic_target, notes_depth, active_key)
                    st.session_state["active_notes"] = notes_output
                    st.session_state["active_topic"] = topic_target

    # Clean Output Render (No weird empty box, direct PDF download)
    if "active_notes" in st.session_state:
        st.markdown("---")
        action_col1, action_col2 = st.columns([3, 1])
        with action_col2:
            pdf_bytes = create_pdf_from_text(st.session_state['active_topic'], st.session_state["active_notes"])
            st.download_button(
                label="📥 Download Notes (PDF)",
                data=pdf_bytes,
                file_name=f"{st.session_state['active_topic'].lower().replace(' ', '_')}_notes.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        st.markdown(st.session_state["active_notes"])

# TAB 2: SMART QUIZ
elif st.session_state["active_tab"] == TAB_QUIZ:
    st.markdown("""
    <div class="glass-panel">
        <h4 style="margin: 0 0 6px 0; font-weight: 700; color: #f8fafc;">Self-Assessment Interactive Quiz</h4>
        <p style="margin: 0 0 16px 0; font-size: 0.88rem; color: #94a3b8;">
            Upload your textbook PDF and test your knowledge with immediate scoring and explanations.
        </p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1])
    with col1:
        pdf_quiz = st.file_uploader("Upload Textbook Chapter (PDF)", type=["pdf"], key="pdf_quiz")
    with col2:
        topic_quiz = st.text_input("Quiz Topic", placeholder="e.g. Guided Media", key="topic_quiz")
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("⚡ Generate Interactive Quiz", type="primary", use_container_width=True):
        if not pdf_quiz:
            st.error("Please upload PDF first.")
        elif not topic_quiz.strip():
            st.error("Please provide topic's name.")
        else:
            try:
                start_p, end_p = map(int, [x.strip() for x in page_range_val.split("-")])
            except ValueError:
                start_p, end_p = 1, 100

            active_key = user_key_input.strip() or get_system_api_key()

            with st.spinner("Formulating multiple-choice questions from your textbook..."):
                blocks = extract_meaningful_pdf_blocks(pdf_quiz.read(), start_p, end_p)
                if not blocks:
                    st.error("Unable to extract PDF.")
                else:
                    context = rank_and_retrieve_context(blocks, topic_quiz, top_k=12)
                    quiz_res = generate_interactive_quiz(context, topic_quiz, active_key)
                    if quiz_res:
                        st.session_state["quiz_data"] = quiz_res
                        st.session_state["quiz_submitted"] = False
                    else:
                        st.error("Issue in generating Quiz. Please provide Groq API key.")

    if "quiz_data" in st.session_state and st.session_state["quiz_data"]:
        st.markdown("---")
        st.markdown("### 📝 Test Your Knowledge")
        
        user_responses = {}
        for idx, q_item in enumerate(st.session_state["quiz_data"]):
            st.markdown(f"**Question {idx + 1}: {q_item['question']}**")
            user_responses[idx] = st.radio(
                "Options:",
                q_item["options"],
                key=f"quiz_opt_{idx}",
                index=None
            )
            st.write("")

        if st.button("✅ Submit Answers", type="primary"):
            st.session_state["quiz_submitted"] = True

        if st.session_state.get("quiz_submitted"):
            st.markdown("---")
            score = 0
            total_q = len(st.session_state["quiz_data"])
            
            for idx, q_item in enumerate(st.session_state["quiz_data"]):
                selected = user_responses.get(idx)
                if selected == q_item["answer"]:
                    score += 1
                    st.success(f"**Q{idx + 1}: Correct!** {q_item['explanation']}")
                else:
                    st.error(f"**Q{idx + 1}: Incorrect.** Correct Answer: **{q_item['answer']}**\n\n*Explanation:* {q_item['explanation']}")

            st.metric(label="🏆 Final Score", value=f"{score} / {total_q}", delta=f"{int((score/total_q)*100)}% Accuracy")

# TAB 3: INPAINTER
else:
    st.markdown("""
    <div class="glass-panel">
        <h4 style="margin: 0 0 6px 0; font-weight: 700; color: #f8fafc;">Document Stroke Cleaner & Inpainter</h4>
        <p style="margin: 0 0 16px 0; font-size: 0.88rem; color: #94a3b8;">
            Remove handwritten doodles, marker highlights, or scan artifacts from textbook pages with OpenCV Telea inpainting.
        </p>
    """, unsafe_allow_html=True)

    clean_file = st.file_uploader("Upload Scanned Page Image", type=["png", "jpg", "jpeg", "webp"], key="clean_uploader")
    st.markdown("</div>", unsafe_allow_html=True)

    if clean_file:
        file_bytes = np.asarray(bytearray(clean_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_bgr.shape[:2]

        canvas_w = 760
        canvas_h = int(h * (canvas_w / w))
        img_resized = cv2.resize(img_rgb, (canvas_w, canvas_h))

        if not CANVAS_AVAILABLE:
            st.error("Install `streamlit-drawable-canvas` using: `pip install streamlit-drawable-canvas`")
        else:
            canvas_result = st_canvas(
                fill_color="rgba(239, 68, 68, 0.35)",
                stroke_width=brush_val,
                stroke_color="#EF4444",
                background_image=Image.fromarray(img_resized),
                height=canvas_h,
                width=canvas_w,
                drawing_mode="freedraw",
                return_image_data=True,
                key="doc_canvas_tab"
            )

            if st.button("🧹 Clean Selected Part", type="primary", use_container_width=True):
                if canvas_result is not None and canvas_result.image_data is not None:
                    raw_mask = canvas_result.image_data[:, :, 3]
                    if np.max(raw_mask) == 0:
                        st.info("Please mark the area you want to erase.")
                    else:
                        mask_full = cv2.resize(raw_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                        _, bin_mask = cv2.threshold(mask_full, 10, 255, cv2.THRESH_BINARY)

                        cleaned_bgr = clean_document_mask(img_bgr, bin_mask.astype(np.uint8), inpaint_radius_val)
                        cleaned_rgb = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2RGB)

                        col_l, col_r = st.columns(2)
                        with col_l:
                            st.image(img_rgb, caption="Original Document", use_container_width=True)
                        with col_r:
                            st.image(cleaned_rgb, caption="Restored Output", use_container_width=True)

                        _, buf = cv2.imencode(".png", cleaned_bgr)
                        st.download_button(
                            "📥 Download Restored Page",
                            buf.tobytes(),
                            "restored_document.png",
                            "image/png",
                            use_container_width=True
                        )