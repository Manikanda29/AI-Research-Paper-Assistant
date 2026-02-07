from openai import OpenAI
import streamlit as st

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def summarize(text):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Summarize academic text clearly"},
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message.content
# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="AI Research Paper Assistant",
    layout="wide"
)
st.markdown("""
<style>
button[data-baseweb="tab"] {
    padding: 6px 12px !important;
    font-size: 14px !important;
    height: 40px !important;
}

div[data-baseweb="tab-list"] {
    gap: 6px;
}

div[data-baseweb="tab-panel"] {
    padding-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# LOAD AI MODEL (ONCE)
# ===============================
from transformers import pipeline
import streamlit as st

@st.cache_resource
def load_summarizer():
    return pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        device=-1  # force CPU
    )
summarizer = load_summarizer()

# ===============================
# UTIL FUNCTIONS
# ===============================
def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()

def extract_text_from_pdf(uploaded_pdf):
    doc = fitz.open(stream=uploaded_pdf.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text.strip()

def extract_keywords(text, top_n=8):
    vec = TfidfVectorizer(stop_words="english", max_features=top_n)
    vec.fit([text])
    return vec.get_feature_names_out()

def predict_user_intent(keywords):
    k = [x.lower() for x in keywords]
    if any(x in k for x in ["survey", "review"]):
        return "📚 Survey / Review focused research"
    if any(x in k for x in ["model", "network", "algorithm"]):
        return "🧠 Model / Algorithm based research"
    if any(x in k for x in ["dataset", "experiment", "data"]):
        return "📊 Dataset / Experimental research"
    return "🔍 General research exploration"

def typing_effect(text):
    box = st.empty()
    output = ""
    for ch in text:
        output += ch
        box.markdown(output)
        time.sleep(0.002)

# ===============================
# ARXIV SEARCH
# ===============================
def search_arxiv(query, max_results=50):
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results}"
    return requests.get(url).text

def parse_arxiv(feed):
    papers = []
    entries = feed.split("<entry>")
    for entry in entries[1:]:
        title = entry.split("<title>")[1].split("</title>")[0].strip()
        link = entry.split("<id>")[1].split("</id>")[0].strip()
        papers.append((title, link))
    return papers

# ===============================
# HERO SECTION
# ===============================
st.markdown("""
<h1 style='text-align:center;'>🤖 AI Research Paper Assistant</h1>
<p style='text-align:center;color:gray;'>
Summarize • Discover • Search research papers intelligently
</p>
""", unsafe_allow_html=True)

# ===============================
# INPUT SECTION
# ===============================
st.markdown("### 📄 Upload or Paste Research Content")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

text_input = ""
if uploaded_file:
    with st.spinner("Extracting text from PDF..."):
        text_input = extract_text_from_pdf(uploaded_file)
        st.success("PDF content extracted ✅")

text_input = st.text_area(
    "Or paste research text",
    value=text_input,
    height=180,
    placeholder="Paste abstract / introduction here..."
)

input_ready = len(text_input.strip()) > 50

st.divider()

# ===============================
# TABS
# ===============================
tab1, tab2, tab3 = st.tabs([
    "📝 Generate Summary",
    "📚 Suggest Related Papers",
    "🔍 Search Research Papers"
])

# ===============================
# TAB 1 — SUMMARY
# ===============================
with tab1:
    st.subheader("Generate AI Summary")

    length = st.radio(
        "Summary Length",
        ["Short", "Medium", "Long"],
        horizontal=True
    )

    if st.button("✨ Generate Summary", disabled=not input_ready):
        content = clean_text(text_input)

        if len(content) < 200:
            st.error("Please upload a proper research PDF or paste more text (min 200 characters).")
            st.stop()

        content = content[:2000]

        max_len = {"Short": 80, "Medium": 140, "Long": 220}[length]

        with st.status("🤖 AI is analyzing the paper...", expanded=True) as status:
            time.sleep(0.8)
            st.write("📄 Reading content")
            time.sleep(0.8)
            st.write("🧠 Understanding context")
            time.sleep(0.8)
            st.write("✍️ Generating summary")

            result = summarizer(
                content,
                max_length=max_len,
                min_length=int(max_len * 0.4),
                do_sample=False
            )

            status.update(label="✅ Summary generated", state="complete")

        with st.expander("📄 View Summary", expanded=True):
            typing_effect(result[0]["summary_text"])

        keywords = extract_keywords(content)
        st.info("🔑 Keywords: " + ", ".join(keywords))
        st.success(predict_user_intent(keywords))

# ===============================
# TAB 2 — RELATED PAPERS
# ===============================
with tab2:
    st.subheader("AI Suggested Related Papers")

    num_papers = st.number_input(
        "Enter how many papers you want",
        min_value=1,
        max_value=50,
        value=10,
        step=1,
        key="related_papers_input"
    )

    if st.button("📚 Suggest Papers", disabled=not input_ready):
        with st.spinner("Finding related research papers..."):
            content = clean_text(text_input)
            keywords = extract_keywords(content)
            query = " ".join(keywords[:3])

            feed = search_arxiv(query)
            papers = parse_arxiv(feed)

            for i, (title, link) in enumerate(papers[:num_papers], 1):
                with st.expander(f"{i}. {title}"):
                    st.markdown(f"[📄 Read Paper]({link})")

# ===============================
# TAB 3 — SEARCH PAPERS
# ===============================
with tab3:
    st.subheader("Search Research Papers")

    num_papers = st.number_input(
        "Enter how many papers you want",
        min_value=1,
        max_value=50,
        value=10,
        step=1,
        key="search_papers_input"
    )

    query = st.text_input(
        "Search research paper",
        placeholder="e.g. smart autonomous drone"
    )

    if query:
        with st.spinner("Searching papers..."):
            feed = search_arxiv(query, max_results=int(num_papers))
            papers = parse_arxiv(feed)

            st.info("AI Intent: " + predict_user_intent(query.split()))

            for i, (title, link) in enumerate(papers, 1):
                st.markdown(f"**{i}. {title}**")
                st.markdown(f"[Read Paper]({link})")

# ===============================

st.caption("🚀 Built by Manikandan S | AI Research Paper Assistant")

