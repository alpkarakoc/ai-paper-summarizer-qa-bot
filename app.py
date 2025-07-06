import streamlit as st
from summarizer import PaperSummarizer
from qa_module import QuestionAnsweringBot
from pdf_parser import extract_text_from_pdf
from embeddings import EmbeddingRetriever

st.set_page_config(page_title="AI Paper Summarizer & QA Bot", layout="wide")

st.title("📚 AI-Powered Scientific Paper Summarizer & Q&A Bot")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    # Save uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Extract text from PDF
    raw_text = extract_text_from_pdf("temp.pdf")

    # Summarize the paper
    with st.spinner("Summarizing the paper..."):
        summarizer = PaperSummarizer()
        summary = summarizer.summarize(raw_text)
    st.subheader("🔍 Summary")
    st.write(summary)

    # Build embeddings
    with st.spinner("Preparing context for Q&A..."):
        retriever = EmbeddingRetriever()
        retriever.build_vectorstore(raw_text)

    # QA interaction
    st.subheader("💬 Ask a question about the paper:")
    user_question = st.text_input("Type your question here")

    if user_question:
        context = retriever.retrieve_relevant_context(user_question)
        qa_bot = QuestionAnsweringBot()
        answer = qa_bot.answer_question(user_question, context)
        st.markdown(f"**Answer:** {answer}")
