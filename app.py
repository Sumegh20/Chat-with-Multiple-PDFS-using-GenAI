import streamlit as st
import shutil
from src.helper import save_uploaded_files, get_documnts_from_pdf, text_splitter, create_knowledgebase, get_output, get_output_stream
from params import pdf_folder_path, chunk_size, chunk_overlap, database_path, master_folder
import os
from langchain.schema import AIMessage, HumanMessage

st.set_page_config("Chat PDF")
st.header("Chat with your PDF/s")

with st.sidebar:
    st.title("Menu:")
    pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)

    if st.button("Submit & Process"):
        if os.path.exists(master_folder):
            shutil.rmtree(master_folder)
        if pdf_docs:
            with st.spinner("Processing..."):
                num_files = save_uploaded_files(pdf_docs, pdf_folder_path)
                # st.success(f"Recieved {num_files} PDF file/s")
                document = get_documnts_from_pdf(pdf_folder_path)
                text_chunks = text_splitter(document, chunk_size, chunk_overlap)
                create_knowledgebase(texts=text_chunks, db_path=database_path)

                st.success("Done. Reday for answring you questions.")


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for chat_history in st.session_state.chat_history:
    with st.chat_message(chat_history.role):
        st.markdown(chat_history.content)

if query := st.chat_input("Message"):
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        stream = get_output_stream(query, st.session_state.chat_history)
        response = st.write_stream(stream)

        # For static messages
        # response = get_output(query, st.session_state.chat_history)
        # st.write(response)
        
    st.session_state.chat_history.extend([
                                    HumanMessage(role="user", content=query),
                                    AIMessage(role="assistant", content=response)# stream),
                                    ])