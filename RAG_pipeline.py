# -*- coding: utf-8 -*-
"""Copy of InternTask_Section1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/12vqFMtNtvtd__8SzOjaJWJmlMtl5EXqT

RAG PIPELINE using Elastic Search indexing Framework
"""

pip install elasticsearch_serverless accelerate groq

"""Data Preparation"""

import json
import pandas as pd


with open('/content/Amazon_sagemaker_Faq.txt','r') as file:
    data=json.load(file)

df=pd.DataFrame(data)

"""Indexing"""

from elasticsearch_serverless import Elasticsearch
import random
es = Elasticsearch(
    "https://c4496a6433094eb5a85e706f3cb47be5.es.us-east-1.aws.elastic.cloud:443",
    api_key="Uk1SSFlvOEJjdnMxa0Z5Y1pjQ3o6Vjl4cDZWbERRRVdNX0xoSGxGYTRHdw==")
index_name=str(random.randint(1,100))

def index_documents(df):

    es.indices.create(index=index_name)

    for index, row in df.iterrows():
      es.index(index=index_name, id=index, body=row.to_dict())

"""Querying Elastic Search"""

def retrieve_documents(query):
    res = es.search(index=index_name, body={'query': {'match': {'question': query}}})
    hits = res['hits']['hits']
    return [hit['_source'] for hit in hits]

"""Text Generation"""

from groq import Groq


client = Groq(api_key="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")

def generate_answer(documents):
    question= ' '.join([doc['answer'] for doc in documents])
    generated_answer = client.chat.completions.create(
        messages=[
            {
                "role":"user",
                "content":question,
            }],
            model="mixtral-8x7b-32768")
    return generated_answer.choices[0].message.content

"""RAG Pipeline"""

def answer_question(query):
    retrieved_documents = retrieve_documents(query)
    generated_answer = generate_answer(retrieved_documents)
    return generated_answer

"""Answering Questions"""

index_documents(df)
try:
    query = input("Ask your question here:")
    answer = answer_question(query)
    print("Answer:", answer)
except:
    print("Bad request, Not found")