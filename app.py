# import streamlit as st
import streamlit as st
from Rag import RAG_retrival,EmbeddingManager,VectorStoreManager,load_llm,generate_output
import pickle
import re
import pandas as pd
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
load_dotenv()
GROQ_API = os.getenv("GROQ_API_KEY")

# obj_doct = df_to_text()
# obj_chunk = doct_to_chunk(obj_doct)
embedd = EmbeddingManager()

vector_store = VectorStoreManager()
# text = [i.page_content for i in obj_chunk]
# embedding = embedd.chunk_to_embedd(text)
# vector_store.add_document(obj_chunk,embedding)
rag_retrival = RAG_retrival(embedd,vector_store)
llm = load_llm()

df = pickle.load(open("data.pkl","rb"))
# df= data.head()
st.title("Product Recommendation")

query = st.text_input("Enter your query")

if st.button("Search"):

    respone = rag_retrival.retrival(query)

    answer = generate_output(query,rag_retrival)
    answer_re_think= re.sub(r'<think>.*?</think>','',answer,flags=re.DOTALL).strip()
    st.write(answer_re_think)

    # st.write(respone)
   
    # id = []
    #    # uniq_id =[]
    # for item in respone:
    #     # match = re.search(r"pid:([^,]+)",item["document"])
    #     # if not match:
    #     #     match = re.search(r"uniq_id:([^,]+)",item["document"])
    #     # # uniq_id.append(match.group(1))
    #     # if match:
    #     #     id.append(match.group(1))
        
    #     if "pid" in item:
    #         doc_id = item["pid"]
    #         id.append(doc_id)
    #     else:
    #         doc_id = item.get("metadata", {}).get("uniq_id")
    #         id.append(doc_id)
    # # st.write(id)
    # for i in id:
    #     product = df[df["uniq_id"] == i]
    #     if not product.empty:
    #         product_name = product["product_name"].iloc[0]
    #         st.write(f"product name : {product_name}")
    #         product_price = product["retail_price"].iloc[0]
    #         st.write(f"product Price : {product_price}")
    #     else:
    #         product = df[df["pid"] == i]
    #         product_name = product["product_name"].iloc[0]
    #         st.write(f"product name : {product_name}")
    #         product_price = product["retail_price"].iloc[0]
    #         st.write(f"product Price : {product_price}")
        # for i in id:
        #    product = df[df["uniq_id"] == i]
        #    if not product.empty:
        #        st.write(product["product_name"].iloc[0])
        #        st.write(product["retail_price"].iloc[0])
        #    else:
        #        product = df[df["pid"] == i]
        #        st.write(product["product_name"].iloc[0])
        #        st.write(product["retail_price"].iloc[0])
       
        # st.write("Document ID:", result["id"])

        # product = df[
        #     (df["pid"] == result["metadata"].get("product_id", ""))
        # ]

        # if not product.empty:
        #     st.write("Product Name:", product["product_name"].iloc[0])
        #     st.write("Price:", product["discounted_price"].iloc[0])

        