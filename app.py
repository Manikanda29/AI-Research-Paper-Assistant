import streamlit as st
import re, time, fitz, requests
from sklearn.feature_extraction.text import TfidfVectorizer
from openai import OpenAI

# ===============================
# OPENAI CLIENT
# ===============================
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def summarize(text, length):
    max_tokens = {"Short": 120, "Medium": 220, "Long": 350}[length]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Summarize academic research text clearly"},
            {"role": "user", "content": text}
        ],
        max_tokens=max_tokens
    )
    return response.choices[0].message.content


# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="AI Research Paper Assistant", layout="wide")

st.markdown("""
<style>
button[data-baseweb="tab"] {
    padding: 6px 12px !important;
    font-size: 14px !important;
    height: 40px !important;
}
div[data-baseweb="tab-list"] { gap: 6px; }
</style>
""", unsafe_allow_html=True)


# ===============================
# UTIL FUNCTIONS
# ===============================
def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()

def extract_text_from_pdf(uploaded_pdf):
    doc = fitz.open(stream=uploaded_pdf.read(), filetype="pdf")
    return " ".join(page.get_text() for page in doc)

def extract_keywords(text, top_n=8):
    vec = TfidfVectorizer(stop_words="english", max_features=top_n)
    vec.fit([text])
    return vec.get_feature_names_out()

def predict_user_intent(words):
    k = [w.lower() for w in words]
    if any(x in k for x in ["survey", "review"]):
        return "📚 Survey / Review focused research"
    if any(x in k for x in ["model", "algorithm", "network"]):
        return "🧠 Model / Algorithm based research"
    if any(x in k for x in ["dataset", "experiment", "data"]):
        return "📊 Dataset / Experimental research"
    return "🔍 General research exploration"

def typing_effect(text):
    box = st.empty()
    out = ""
    for ch in text:
        out += ch
        box.markdown(out)
        time.sleep(0.002)


# ===============================
# ARXIV SEARCH
# ===============================
def search_arxiv(query, max_results=10):
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&max_results={max_results}"
    return requests.get(url).text

def parse_arxiv(feed):
    papers = []
    for entry in feed.split("<entry>")[1:]:
        title = entry.split("<title>")[1].split("</title>")[0]
        link = entry.split("<id>")[1].split("</id>")[0]
        papers.append((title.strip(), link.strip()))
    return papers


# ===============================
# UI
# ===============================
st.markdown("<h1 style='text-align:center;'>🤖 AI Research Paper Assistant</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
text_input = extract_text_from_pdf(uploaded_file) if uploaded_file else ""

text_input = st.text_area("Or paste research text", value=text_input, height=180)
ready = len(text_input.strip()) > 200

tab1, tab2, tab3 = st.tabs(["📝 Generate Summary", "📚 Suggest Papers", "🔍 Search Papers"])


# ===============================
# TAB 1
# ===============================
with tab1:
    length = st.radio("Summary Length", ["Short", "Medium", "Long"], horizontal=True)

    if st.button("✨ Generate Summary", disabled=not ready):
        content = clean_text(text_input)[:3000]

        with st.spinner("AI summarizing..."):
            summary = summarize(content, length)

        typing_effect(summary)

        keywords = extract_keywords(content)
        st.info("🔑 Keywords: " + ", ".join(keywords))
        st.success(predict_user_intent(keywords))


# ===============================
# TAB 2
# ===============================
with tab2:
    if st.button("📚 Suggest Related Papers", disabled=not ready):
        keywords = extract_keywords(text_input)
        feed = search_arxiv(" ".join(keywords[:3]))
        for i, (t, l) in enumerate(parse_arxiv(feed), 1):
            st.markdown(f"**{i}. {t}**  \n[Read Paper]({l})")


# ===============================
# TAB 3
# ===============================
with tab3:
    query = st.text_input("Search topic")
    if query:
        feed = search_arxiv(query)
        for i, (t, l) in enumerate(parse_arxiv(feed), 1):
            st.markdown(f"**{i}. {t}**  \n[Read Paper]({l})")


st.caption("🚀 Built by Manikandan S | AI Research Paper Assistant")
