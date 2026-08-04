import streamlit as st
from ingest import search
from generate import generate_answer

st.set_page_config(page_title="Hospital Chatbot", page_icon="🏥")
st.title("Hospital Chatbot")
st.caption("Ask a question about your patient record.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

question = st.chat_input("Type your question here")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Looking up your record..."):
            results = search(question, n_results=2)
            retrieved_docs = results["documents"][0]
            answer = generate_answer(question, retrieved_docs)
        st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})