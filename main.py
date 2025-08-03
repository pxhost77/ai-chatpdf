__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# from dotenv import load_dotenv
# load_dotenv()
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA
import streamlit as st
import tempfile
import os
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from streamlit_extras.buy_me_a_coffee import button
button(username="nevido", floating=True, width=221)

# from langchain_openai import OpenAIEmbeddings
# help(OpenAIEmbeddings)

st.title("ChatPDF")
st.write("---")

# OpenAI KEY 입력 받기
my_key = st.text_input('Enter your OpenAI API Key', type="password")

if my_key:
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=my_key)
        st.success("✅ API 키가 정상적으로 적용되었습니다.")
    except Exception as e:
        st.error(f"❌ 오류 발생: {e}")
        

# 파일 업로드

uploaded_file = st.file_uploader("PDF 파일을 올려주세요", type=['pdf'])
st.write("---")


def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath,"wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

if uploaded_file is not None:
    with st.spinner("📖 PDF 파일 처리 중..."):
        pages = pdf_to_document(uploaded_file)
#업로드되면 동작하는 코드

if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)
        
#Loader
# loader = PyPDFLoader("unsu.pdf")
# pages = loader.load_and_split()
                              
    #split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size= 300,
        chunk_overlap = 20,
        length_function = len,
        is_separator_regex=False,
    )
    texts = text_splitter.split_documents(pages)

    # Embedding
    embedding_model = OpenAIEmbeddings(api_key=my_key)

    # load it into Chrom
    db = Chroma.from_documents(texts,embedding_model)
    
    # Stream 받아줄 핸들러 만들기
    from langchain_core.callbacks.base import BaseCallbackHandler
    
    class StreamHandler(BaseCallbackHandler):
        def __init__(self,  container, initial_text=""):
            self.container = container
            self.text=initial_text
        def on_llm_new_token(self, token: str, **kwargs) -> None:
            self.text+=token
            self.container.markdown(self.text)
        

    #  Question
    st.header("PDF에게 질문해보세요!")
    question = st.text_input("질문을 입력하세요")
    #st.write("The current movie title is", title)
    
    
    if st.button("질문하기"):
        with st.spinner('Wait for it...'):

    #question = "아내가 먹고 싶어하는 음식은 무엇이야?"
    # llm = ChatOpenAI(temperature=0)
    # retriver_from_llm = MultiQueryRetriever.from_llm(
    #     retriever=db.as_retriever(), llm=llm
    # )
    # docs = retriver_from_llm.get_relevant_documents(query=question)
    # print(len(docs))
    # print(docs)
            chat_box = st.empty()
            stream_hander = StreamHandler(chat_box)
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0,api_key=my_key,streaming=True,callbacks=[stream_hander])
            qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())
            qa_chain({"query": question})
            # print(result)
            # st.write(result["result"])


