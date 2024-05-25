from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain_community.chat_models import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
from params import *

load_dotenv()
# os.environ["OPENAI_API_KEY"] = os.environ.get('OPENAI_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.environ.get('GOOGLE_API_KEY')
os.environ['GROQ_API_KEY'] = os.environ.get('GROQ_API_KEY')

# Read multiple pdf files
def get_documnts_from_pdf(folder_path):
    loader = PyPDFDirectoryLoader(folder_path)
    data = loader.load()

    return data

def save_uploaded_files(uploaded_files, folder_path):
    os.makedirs(folder_path, exist_ok=True)
    for uploaded_file in uploaded_files:
        file_path = os.path.join(folder_path, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    return len(uploaded_files)

#Creating text chunks 
def text_splitter(document, chunk_size=500, chunk_overlap=50):
    text_splitter=CharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    chunks=text_splitter.split_documents(document)

    return chunks

#loading embeddings model
def load_embedding():
    # embeddings = OpenAIEmbeddings()
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    return embeddings

# creating the vector database
def create_knowledgebase(texts, db_path="faiss_index"):
    embeddings = load_embedding()

    vectordb = FAISS.from_documents(texts, embeddings)
    vectordb.save_local(db_path)

    # return vectordb

# creating the prompt
def create_prompt():
    template="""You are an assistant for question-answering tasks while prioritizing a seamless user experience.
                Use the following pieces of retrieved context to answer the question.
                If you don't know the answer, just say that you don't know.
                Use ten sentences maximum and keep the answer concise.
                You should be able to remember and reference the last three conversations between you and the user.
                Maintain a friendly, positive, and professional tone throughout interactions.
                Question: {question}
                Context: {context}
                Chat history: {chat_history}
                Answer:    
            """
    
    prompt=ChatPromptTemplate.from_template(template)
    
    return prompt

def get_knowledge_base(db_path, embedding):
    vectordb = FAISS.load_local(db_path, embedding, allow_dangerous_deserialization=True)

    return vectordb

def input_handeler(input: dict):
    return input['question']

# create RAG chain
def create_rag_chain(llm, retriever, prompt):
    rag_chain = (
            # {"context": retriever,  "question": RunnablePassthrough()}
            RunnablePassthrough().assign(
                context = input_handeler | retriever
            )
            | prompt
            | llm
            | StrOutputParser()
        )
    
    return rag_chain

def get_llm():
    
    # OpenAI
    # llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True)

    # Gemini
    # llm = ChatGoogleGenerativeAI(model="gemini-1.0-pro-latest")

    # Groq
    llm = ChatGroq(model='Gemma-7b-It')

    return llm

def get_output(question, chat_history):
    embedding = load_embedding()
    vectordb = get_knowledge_base(db_path=database_path, embedding=embedding)
    llm = get_llm()
    retriever = vectordb.as_retriever()
    prompt = create_prompt()

    rag_chain = create_rag_chain(llm=llm, retriever=retriever, prompt=prompt)
    
    result = rag_chain.invoke({"question": question, "chat_history":chat_history})
    return result

def get_output_stream(question, chat_history):
    embedding = load_embedding()
    vectordb = get_knowledge_base(db_path=database_path, embedding=embedding)
    llm = get_llm()
    retriever = vectordb.as_retriever()
    prompt = create_prompt()

    rag_chain = create_rag_chain(llm=llm, retriever=retriever, prompt=prompt)
    
    result = rag_chain.stream({"question": question, "chat_history":chat_history})
    return result
