from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl

DB_FAISS_PATH = 'vectorstores/db_faiss'

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       #search_kwargs = 2 returns top 2 responses based on input query
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       #injects our custom prompt into the chain
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    #print(qa_chain)
    return qa_chain

#Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model = "llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.5
    )
    return llm

#QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    print(qa)

    return qa

# output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    print(response)
    return response['result']

# response = {'query': 'what is Adrenal gland scan?', 'result': 'Adrenal gland scan is a diagnostic test used to determine if there is a tumor in the adrenal gland. It involves injecting a small amount of a radioactive tracer into a vein, which then collects in the adrenal gland and can be detected by a special camera or scanner. The test can help identify the location and size of a tumor, as well as whether it is secreting excessive amounts of hormones such as adrenaline and noradrenaline.', 'source_documents': [Document(page_content='a tumor in the adrenal gland is suspected. One such situa-\ntion in which a tumor might be suspected is when highblood pressure ( hypertension ) does not respond to med-\nication. Tumors that secrete adrenaline and noradrenalinecan also be found outside the adrenal gland. An adrenalgland scan usually covers the abdomen, chest, and head.\nPrecautions\nAdrenal gland scans are not recommended for preg-', metadata={'source': 'data\\71763-gale-encyclopedia-of-medicine.-vol.-1.-2nd-ed.pdf', 'page': 74}), Document(page_content='GALE ENCYCLOPEDIA OF MEDICINE 2 60Adrenal gland scanGEM - 0001 to 0432 - A  10/22/03 1:41 PM  Page 60', metadata={'source': 'data\\71763-gale-encyclopedia-of-medicine.-vol.-1.-2nd-ed.pdf', 'page': 73})]}

answer = final_result('what is Adrenal gland scan?')
print(answer)

#chainlit code

# This decorator defines a function that gets called when a new chat session starts.
@cl.on_chat_start 
async def start():
    chain = qa_bot()
    # Creates a cl.Message object with a starting message and sends it using await msg.send().
    msg = cl.Message(content="Hello")
    await msg.send()
    msg.content = "Hi, Welcome to Medical Bot. What is your query?"
    await msg.update()
    # Stores the loaded chain (QA model) in the user session using cl.user_session.set("chain", chain). This allows access to the same model throughout the chat session.
    cl.user_session.set("chain", chain) 

# Decorator to react to messages coming from the UI. The decorated function is called every time a new message is received.
@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")

    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]

    if sources:
        answer += f"\n\nSources:" + str(sources)
    else:
        answer += "\nNo sources found"

    await cl.Message(content=answer).send()