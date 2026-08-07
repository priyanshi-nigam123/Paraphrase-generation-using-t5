import streamlit as st
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# ---- Configuration ----
MODEL_REPO = "Priyanshii123/paraphrase-t5"
MAX_LENGTH = 512

st.set_page_config(
    page_title="Paraphrase Generator",
    page_icon="🔁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Custom CSS ----
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #6366F1, #EC4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        color: #6B7280;
        font-size: 1.05rem;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    .paraphrase-card {
        background: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-left: 4px solid #6366F1;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        font-size: 1.02rem;
        line-height: 1.5;
    }
    .original-card {
        background: #EEF2FF;
        border: 1px solid #C7D2FE;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 1.2rem;
        font-size: 1.05rem;
    }
    .stButton>button {
        background: linear-gradient(90deg, #6366F1, #8B5CF6);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.4rem;
        font-weight: 600;
        width: 100%;
    }
    .stButton>button:hover {
        opacity: 0.9;
    }
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

EXAMPLES = [
    "The quick brown fox jumps over the lazy dog.",
    "She enjoys reading books on rainy afternoons.",
    "Climate change is one of the most pressing issues of our time.",
    "The company announced record profits this quarter.",
]


@st.cache_resource(show_spinner=False)
def load_model():
    """Load T5 model and tokenizer from Hugging Face."""
    tokenizer = T5Tokenizer.from_pretrained(MODEL_REPO)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_REPO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model, tokenizer, device


def generate_paraphrase(
    input_text,
    model,
    tokenizer,
    device,
    num_return_sequences=4,
    diversity_penalty=1.5,
):
    """Generate paraphrases for the given input text."""
    processed = "paraphrase: " + input_text

    inputs = tokenizer(
        processed,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=MAX_LENGTH + 20,
            num_beams=4,
            num_return_sequences=num_return_sequences,
            no_repeat_ngram_size=2,
            early_stopping=True,
            temperature=0.9,
            do_sample=True,
            top_p=0.95,
        )

    return [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]


# ---- Initialize Session State ----
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "results" not in st.session_state:
    st.session_state.results = None

# ---- Sidebar Settings ----
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    num_paraphrases = st.slider("Number of Paraphrases", 2, 6, 4)
    diversity_penalty = st.slider("Diversity Level", 0.0, 3.0, 1.5, 0.1)

    st.markdown("---")
    st.markdown("### 💡 Example Sentences")
    for ex in EXAMPLES:
        if st.button(ex, key=f"ex_{ex}", use_container_width=True):
            st.session_state.input_text = ex
            st.session_state.results = None

    st.markdown("---")
    st.caption("Fine-tuned T5 model for English paraphrase generation.")

# ---- Main Header ----
st.markdown('<p class="main-header">🔁 Paraphrase Generator</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Generate multiple natural paraphrases for your sentences using a fine-tuned T5 model.</p>',
    unsafe_allow_html=True,
)

with st.spinner("Loading model... (First load may take a moment)"):
    model, tokenizer, device = load_model()

# ---- Input Section ----
left, right = st.columns([3, 1])

with left:
    input_sentence = st.text_area(
        "Enter Your Sentence",
        value=st.session_state.input_text,
        placeholder="e.g. The quick brown fox jumps over the lazy dog.",
        height=120,
        label_visibility="collapsed",
    )
with right:
    st.write("")
    st.write("")
    generate_clicked = st.button("✨ Generate", use_container_width=True)
    clear_clicked = st.button("Clear", use_container_width=True)

if clear_clicked:
    st.session_state.input_text = ""
    st.session_state.results = None
    st.rerun()

if generate_clicked:
    if not input_sentence.strip():
        st.warning("Please enter a sentence or select an example to begin.")
    else:
        with st.spinner("Generating paraphrases..."):
            st.session_state.results = generate_paraphrase(
                input_sentence,
                model,
                tokenizer,
                device,
                num_return_sequences=num_paraphrases,
                diversity_penalty=diversity_penalty,
            )
        st.session_state.input_text = input_sentence

# ---- Display Results ----
if st.session_state.results:
    st.markdown("### Results")
    st.markdown(
        f'<div class="original-card">📌 <b>Original:</b> {st.session_state.input_text}</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(2)
    for i, para in enumerate(st.session_state.results, 1):
        with cols[(i - 1) % 2]:
            st.markdown(
                f'<div class="paraphrase-card"><b>#{i}</b><br>{para}</div>',
                unsafe_allow_html=True,
            )
else:
    st.info("Enter a sentence above or select an example from the sidebar, then click **Generate** to begin.")
