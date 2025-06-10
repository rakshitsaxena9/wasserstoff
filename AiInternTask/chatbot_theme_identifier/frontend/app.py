import streamlit as st
import requests

BACKEND = "https://wasserstoff-dla4.onrender.com/"
MAX_DOCS = 75

st.set_page_config("GenAI Doc QA", layout="centered")
st.title("GenAI Document QA & Theme Chatbot")

# ===== State Initialization =====
if 'history' not in st.session_state:
    st.session_state['history'] = []
if 'uploaded_files' not in st.session_state:
    st.session_state['uploaded_files'] = set()
if 'uploaded_any' not in st.session_state:
    st.session_state['uploaded_any'] = False
if 'upload_disabled' not in st.session_state:
    st.session_state['upload_disabled'] = False
if 'chat_input' not in st.session_state:
    st.session_state['chat_input'] = ""

# ===== Document Upload Section =====
st.sidebar.header("1. Upload Documents (one-time)")
if not st.session_state['uploaded_any']:
    uploaded_files = st.sidebar.file_uploader(
        f"Upload up to {MAX_DOCS} files (PDF, text, images)",
        type=["pdf", "txt", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="file_uploader",
        disabled=st.session_state['upload_disabled']
    )
    if uploaded_files:
        if len(uploaded_files) > MAX_DOCS:
            st.sidebar.error(f"Cannot upload more than {MAX_DOCS} documents.")
        else:
            if st.sidebar.button("Confirm Upload", key="confirm_upload"):
                all_uploaded = True
                for f in uploaded_files:
                    if f.name not in st.session_state['uploaded_files']:
                        files = {"file": (f.name, f, f.type)}
                        with st.spinner(f"Uploading {f.name}..."):
                            resp = requests.post(f"{BACKEND}/upload/", files=files)
                            res = resp.json()
                        if res.get("success"):
                            st.session_state['uploaded_files'].add(f.name)
                        else:
                            st.sidebar.error(f"Failed: {res.get('error')}")
                            all_uploaded = False
                    else:
                        st.sidebar.info(f"{f.name} already uploaded.")
                if all_uploaded:
                    st.sidebar.success(f"Uploaded {len(uploaded_files)} document(s).")
                    st.session_state['uploaded_any'] = True
                    st.session_state['upload_disabled'] = True
else:
    st.sidebar.info(f"Uploaded {len(st.session_state['uploaded_files'])} document(s). Upload is disabled.")

if not st.session_state['uploaded_any']:
    st.sidebar.warning("Please upload and confirm your documents before asking questions.")

st.sidebar.markdown("---")

# ===== Conversational Chat History =====
st.markdown("### Conversation")
for item in st.session_state['history']:
    # User message
    st.markdown(
        f"<span style='color:#41c9ff'><b>You:</b> {item['question']}</span>",
        unsafe_allow_html=True
    )

    # Grouped block for AI reply (table + answer)
    with st.container():
        st.markdown(
            """
            <div style="background-color: #181a20; border-radius: 14px; padding: 16px 16px 8px 16px; margin: 0.5em 0 1.3em 0; box-shadow: 0 2px 12px #0002;">
            """,
            unsafe_allow_html=True
        )
        if item['answers']:
            table_data = [
                {
                    "Index": idx + 1,
                    "Document Name": a.get("file_name", a.get("doc_id")),
                    "Answer": a["answer"],
                    "Citation": a["citation"]
                }
                for idx, a in enumerate(item["answers"])
            ]
            st.table(table_data)
        if item['themes']:
            st.markdown(
                f"<div style='color:#fff; margin-top:0.4em;'><b>AI:</b> {item['themes']}</div>",
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")

if not st.session_state['uploaded_any']:
    st.info("Upload and confirm documents to start chatting.")

# ===== Chat Input Section =====
def send_message():
    user_query = st.session_state['chat_input'].strip()
    if user_query:
        with st.spinner("Getting answer..."):
            resp = requests.post(f"{BACKEND}/query/", data={"user_query": user_query})
            result = resp.json()
        chat_item = {
            "question": user_query,
            "answers": result.get("answers", []),
            "themes": result.get("themes", "")
        }
        st.session_state['history'].append(chat_item)
        st.session_state['chat_input'] = ""  # This works in the callback only

if st.session_state['uploaded_any']:
    st.markdown("---")
    st.text_input(
        "Type your question and press Enter to send",
        key="chat_input",
        label_visibility="collapsed",
        on_change=send_message
    )
else:
    st.info("Please upload and confirm your documents to start chatting.")

st.sidebar.markdown("---")
st.sidebar.caption("Built with Gemini, ChromaDB, and Streamlit")

