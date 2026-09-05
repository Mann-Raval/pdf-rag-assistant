import os
import tempfile
import streamlit as st
from rag import process_pdf, get_answer

# ── PAGE CONFIG ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📄",
    layout="wide"
)

# ── SESSION STATE (Streamlit's memory between reruns) ────────────────
# Every time user does anything, Streamlit reruns the whole script.
# session_state variables SURVIVE those reruns — like global variables.
if "messages" not in st.session_state:
    st.session_state.messages = []      # chat history

if "retriever" not in st.session_state:
    st.session_state.retriever = None   # RAG retriever after PDF upload

if "pdf_ready" not in st.session_state:
    st.session_state.pdf_ready = False  # tracks if PDF is processed

# ── SIDEBAR ──────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📄 PDF RAG Assistant")
    st.caption("Your AI knowledge assistant")
    st.divider()

    # PDF Upload
    uploaded_file = st.file_uploader(
        "Upload your PDF",
        type=["pdf"],               # only allow PDF files
        accept_multiple_files=False # one file at a time
    )

    if uploaded_file is not None:
        # Reset if different file uploaded
        if st.session_state.get("current_pdf") != uploaded_file.name:
            st.session_state.pdf_ready = False
            st.session_state.messages = []
            st.session_state.current_pdf = uploaded_file.name

        if not st.session_state.pdf_ready:
            with st.spinner("Processing PDF..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                st.session_state.retriever = process_pdf(tmp_path)
                st.session_state.pdf_ready = True
                os.unlink(tmp_path)
            st.success("✅ PDF Ready!")

    # Show PDF status
    if st.session_state.pdf_ready:
        st.info(f"📄 {uploaded_file.name} loaded")

    st.divider()

    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── MAIN AREA ────────────────────────────────────────────────────────
st.title("PDF RAG Assistant")
st.caption("Ask questions, extract details, or summarize your uploaded PDF.")

# Show suggestion cards if no messages yet
if len(st.session_state.messages) == 0:
    st.write("") # spacing
    col1, col2 = st.columns(2)

    with col1:
        if st.button("📋 Summarize the document", use_container_width=True):
            st.session_state.messages.append({
                "role": "user",
                "content": "Summarize the document"
            })
            st.rerun()

        if st.button("📚 Show the main topics", use_container_width=True):
            st.session_state.messages.append({
                "role": "user",
                "content": "What are the main topics covered?"
            })
            st.rerun()

    with col2:
        if st.button("💡 What are the key takeaways?", use_container_width=True):
            st.session_state.messages.append({
                "role": "user",
                "content": "What are the key takeaways?"
            })
            st.rerun()

        if st.button("🔍 Explain the core concept", use_container_width=True):
            st.session_state.messages.append({
                "role": "user",
                "content": "Explain the core concept of this document"
            })
            st.rerun()

# ── DISPLAY CHAT HISTORY ─────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── HANDLE NEW QUESTION ──────────────────────────────────────────────
# This handles BOTH typed questions AND suggestion card clicks
if len(st.session_state.messages) > 0:
    last_message = st.session_state.messages[-1]

    # If last message is from user and has no answer yet
    if last_message["role"] == "user":
        if st.session_state.retriever is None:
            st.warning("⚠️ Please upload a PDF first!")
        else:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = get_answer(
                        last_message["content"],
                        st.session_state.retriever
                    )
                st.markdown(answer)

            # Save assistant response to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

# ── CHAT INPUT ───────────────────────────────────────────────────────
user_input = st.chat_input("Ask your PDF anything...")

if user_input:
    # Add user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    st.rerun()