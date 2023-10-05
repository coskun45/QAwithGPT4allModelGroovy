from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import GPT4All, LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pdf2image import convert_from_path
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

import os
import argparse
import time

load_dotenv()

model_paths=os.environ.get('MODEL_PATH')
model_n_ctx=os.environ.get('MODEL_N_CTX')
model_n_batch=os.environ.get('MODEL_N_BATCH',8)
target_source_chunks=int(os.environ.get('TARGET_SOURCE_CHUNKS',4))

def main():
    args=parse_arguments()
    FAISS_INDEX_PATH="./faiss-index-250"
    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    faiss_index=FAISS.load_local(FAISS_INDEX_PATH,embeddings)
    retriever=faiss_index.as_retriever(search_kwargs={"k":target_source_chunks})
    callbacks=[] if args.mute_stream else [StreamingStdOutCallbackHandler()]

    llm=GPT4All(model=model_paths,max_tokens=1000,backend='gptj',n_batch=model_n_batch,callbacks=callbacks,verbose=False)
    qa=RetrievalQA.from_chain_type(llm=llm,chain_type="stuff",retriever=retriever,return_source_documents=not args.hide_source)

    while True:
        query=input("\n Enter a query: ")
        if query =="exit":
            break
        if query.strip()=="":
            continue

        start=time.time()
        res=qa(query)
        answer,docs=res['result'],[] if args.hide_source else res['source_documents']
        end=time.time()

        print("\n\n> Question: ")
        print(query)
        print(f"\n> Ansert (took {round(end-start,2)}s.): ")
        print(answer)


def parse_arguments():
    parser=argparse.ArgumentParser(description='privateGPT: Ask questions on the document privately, ' 'using the power of LLMs and gpt4all.')

    parser.add_argument("--hide-source", "-S",
                        action='store_true', 
                        help='Use tis flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='User this flag to disable the streaming StdOut callback for LLMs.')
    
    return parser.parse_args()

if __name__=="__main__":
    main()
