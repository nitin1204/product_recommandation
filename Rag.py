#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
from langchain_community.document_loaders.text import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import uuid
# import streamlit as st


# In[ ]:


# load data
df = pd.read_csv("flipkard_product_data.csv")


# In[3]:


# handal mising valuse
df["retail_price"] = df["retail_price"].fillna('')
df["discounted_price"] = df["discounted_price"].fillna('')
df["image"] = df["image"].fillna('')
df["description"] = df["description"].fillna('')
df["brand"] = df["brand"].fillna('')
df["product_specifications"] = df["product_specifications"].fillna('')


# In[4]:


# df to text
def df_to_text(df):
    texts = []
    for i in range(df.shape[0]):
        row = df.iloc[i]
        text = ""
        for col in df.columns:
            text+= col+":"+str(row[col])+","
        texts.append(text)
    df["text"] = texts
    folder_name = "data"
    os.makedirs(folder_name,exist_ok=True)
    file_name = os.path.join(folder_name,"product_data.txt")
    with open(file_name,"w",encoding="utf-8") as f:
        for line in df["text"]:
            f.write(line+"\n")
text = df_to_text(df)


# In[5]:


# text to document
def text_to_document():
  doct_loader = TextLoader("data/product_data.txt",encoding="utf-8")
  doct = doct_loader.load()
  return doct
obj_doct = text_to_document()


# In[ ]:
from langchain_core.documents import Document

def dataframe_to_documents(df):
    documents = []

    for _, row in df.iterrows():

        text = ""
        for col in df.columns:
            text += f"{col}:{row[col]},"

        doc = Document(
            page_content=text,
            metadata={
                "pid": str(row["pid"]),
                "uniq_id": str(row["uniq_id"]),
                "product_name": str(row["product_name"])
            }
        )

        documents.append(doc)

    return documents


obj_doct = dataframe_to_documents(df)


# doct to chunk
def doct_to_chunk(document,chunk_size=300,chunk_overlap=30):
    chunk_loader = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )
    chunk = chunk_loader.split_documents(document)
    return chunk
obj_chunk = doct_to_chunk(obj_doct)


# In[7]:

# @st.cache_resource
# chunk to embedding
class EmbeddingManager():
    def __init__(self,model_name ="all-MiniLM-L6-v2"):
        self.model_name = model_name
        print("Loadding model..",self.model_name)
        self.model = SentenceTransformer(self.model_name)
        print("Embedding dimension",self.model.get_sentence_embedding_dimension())

    def chunk_to_embedd(self,text):
        embedding = self.model.encode(text,show_progress_bar=True)
        print("Embedding Shape",embedding.shape)
        return embedding
embedd = EmbeddingManager()


# In[16]:

# @st.cache_resource
# embedding to vector store
class VectorStoreManager:
    def __init__(self,persist_directory="data/vector_data",collection_name="text_document"):
        self.presist_directory = persist_directory
        self.collection_name = collection_name
        self.cilent = None
        self.collection = None
        self.initialize_score()
    def initialize_score(self):
        os.makedirs(self.presist_directory,exist_ok=True)
        # create cilent
        self.cilent = chromadb.PersistentClient(path=self.presist_directory)
        # create collection
        self.collection = self.cilent.get_or_create_collection(
            name = self.collection_name,
            metadata={"description":"vector store collection for text embedding in RAG"}


        )
        print("initialize the vector store with collection",self.collection_name)
        print("does in collection",self.collection.count())
    def add_document(self,document,embeddings):
        if len(document) != len(embeddings):
            raise ValueError("num of document does not match num of embedding")
        id = []
        all_metadata = []
        document_content = []
        embedding_list = []

        for i,(doct,embedding) in enumerate(zip(document,embeddings)):

            # doct_id = f"doct_{uuid.uuid4()}"
            # id.append(doct_id)


            # metadata = dict(doct.metadata)
            # metadata["doct_index"] = i
            # metadata["content_leght"] = len(doct.page_content)
            # all_metadata.append(metadata)
            pid = doct.metadata["pid"]
            uniq_id = doct.metadata["uniq_id"]

            doc_id = f"{uniq_id}_{i}"

            id.append(doc_id)

            metadata = {
                "pid": pid,
                "uniq_id": uniq_id,
                "product_name": doct.metadata["product_name"],
                "chunk_index": i,
                "content_length": len(doct.page_content)
            }

            all_metadata.append(metadata)
            document_content.append(doct.page_content)
            embedding_list.append(embedding.tolist())

        max_batch_size = 5000
        for i in range(0,len(id),max_batch_size):
            end = i+max_batch_size
            self.collection.add(
                ids=id[i:end],
                metadatas=all_metadata[i:end],
                embeddings=embedding_list[i:end],
                documents=document_content[i:end]
            )
        print("total document add in vector store",len(document_content))
        print("does in collection",self.collection.count())
vector_store = VectorStoreManager()


# In[17]:


text = [i.page_content for i in obj_chunk]
embedding = embedd.chunk_to_embedd(text)
vector_store.add_document(obj_chunk,embedding)


# In[59]:

# @st.cache_resource
# rag retrival
class RAG_retrival:
    def __init__(self,embedding_model,vector_store):
        self.embedding = embedding_model
        self.vector_data = vector_store
    total_docs = vector_store.collection.count()
    def retrival(self,query,top_k=50):
        query_embedding = self.embedding.chunk_to_embedd(query)
        result = self.vector_data.collection.query(
            query_embeddings = [query_embedding.tolist()],
            n_results= top_k
        )
        retrival_doct = []
        if result["documents"] and result["documents"][0]:
            id = result["ids"][0]
            metadata = result["metadatas"][0]
            document = result["documents"][0]
            distance = result["distances"][0]
            for i,(doct_id,meta,doct,dist) in enumerate(zip(id,metadata,document,distance)):
                retrival_doct.append({
                    "id":doct_id,
                    "metadata":meta,
                    "document":doct,
                    "distance":dist,
                    "rank":i+1
                })
            print(f"retrival {len(retrival_doct)} documents")
        else:
            print("no document found")
        return retrival_doct
rag_retrival = RAG_retrival(embedd,vector_store)




# In[73]:


respone = rag_retrival.retrival("my room size is 30X30 in best bed and price is 30000")
print(respone)

from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
load_dotenv()
GROQ_API = os.getenv("GROQ_API_KEY")


def load_llm():
    return ChatGroq(
        api_key=GROQ_API,
        model = "qwen/qwen3-32b",
        temperature=0.1
    )
llm = load_llm()

def generate_output(query,retrival,top_k=50):
    result = retrival.retrival(query,top_k)
    content = "\n".join(doc["document"] for doc in result) if result else ""
    if not content:
        print("we found no retrival content for the given query")

    prompt = f""" we use given content to generate the answer for the query
            content:{content}
            query:{query} """
    respons = llm.invoke(prompt)
    return respons.content
answer = generate_output("best laptop in dell ",rag_retrival)
print(answer)


# In[74]:


#



# In[ ]:
import pickle 

# save model
pickle.dump(df,open("data.pkl","wb"))

