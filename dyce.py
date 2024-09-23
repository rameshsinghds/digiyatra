
# from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_chroma import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
import pymupdf4llm
import streamlit as st

st.title("Ask your Funds")

@st.cache_resource
def load_and_return_query_engine():
    embed_model = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-large-en-v1.5") #nomic embeddings
    llm = ChatGroq(model="llama3-70b-8192", api_key='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', temperature=0.0, seed=5632)

    faqs = Chroma(embedding_function=embed_model, persist_directory="./faq_db")
    faq_ret = faqs.as_retriever()
    compressor = FlashrankRerank()
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=faq_ret
    )

    prompt = PromptTemplate.from_template(""" {context} 
                                      Answer the following question {query} based on the context provided. 
                                      Do not add phrase like "Based on provided context". 
                                      
                                      If you don't know the answer respond as you dont know. 
                                      """)

    chain = {"context": compression_retriever, "query": RunnablePassthrough()} | prompt | llm | StrOutputParser()

    return chain



if "messages" not in st.session_state:
    st.chat_message("assistant").markdown("Hi! How may I help you today?")
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input

if prompt := st.chat_input("Please type your query here..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            rqe = load_and_return_query_engine()
            st.write_stream(rqe.stream(input=prompt))
            # with st.stdout("info"):
            #     print(response)
            # st.markdown(response)
    # Add assistant response to chat history
    # print(response)
    st.session_state.messages.append({"role": "assistant"})