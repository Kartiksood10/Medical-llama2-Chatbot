from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import chainlit as cl

DB_FAISS_PATH = 'vectorstores/db_faiss'

custom_prompt_template = """Use the following pieces of information and chat history to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
History: {chat_history}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'chat_history', 'question'])
    return prompt

#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):

    memory = ConversationBufferMemory(memory_key="chat_history", input_key="question", output_key='answer', return_messages=True)

    qa_chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       memory = memory,
                                       return_source_documents=True,
                                       get_chat_history= lambda h : h,
                                       combine_docs_chain_kwargs={'prompt': prompt},
                                       verbose = True
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
    response = qa_result({'question': query})
    print(response)
    return response['answer']

# {'question': 'what is Adrenal gland scan?', 'chat_history': [HumanMessage(content='what is Adrenal gland scan?'), AIMessage(content='Adrenal gland scan is a nuclear medicine imaging test that uses small amounts of radioactive materials to visualize the adrenal glands located on top of the kidneys. It can help diagnose conditions such as tumors, infections, or inflammation in the adrenal glands.')], 'answer': 'Adrenal gland scan is a nuclear medicine imaging test that uses small amounts of radioactive materials to visualize the adrenal glands located on top of the kidneys. It can help diagnose conditions such as tumors, infections, or inflammation in the adrenal glands.', 'source_documents': [Document(page_content='a tumor in the adrenal gland is suspected. One such situa-\ntion in which a tumor might be suspected is when highblood pressure ( hypertension ) does not respond to med-\nication. Tumors that secrete adrenaline and noradrenalinecan also be found outside the adrenal gland. An adrenalgland scan usually covers the abdomen, chest, and head.\nPrecautions\nAdrenal gland scans are not recommended for preg-', metadata={'source': 'data\\71763-gale-encyclopedia-of-medicine.-vol.-1.-2nd-ed.pdf', 'page': 74}), Document(page_content='GALE ENCYCLOPEDIA OF MEDICINE 2 60Adrenal gland scanGEM - 0001 to 0432 - A  10/22/03 1:41 PM  Page 60', metadata={'source': 'data\\71763-gale-encyclopedia-of-medicine.-vol.-1.-2nd-ed.pdf', 'page': 73})]}

# answer = final_result('what is Adrenal gland scan?')
# print(answer)

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
    answer = res["answer"]
    sources = res["source_documents"]

    if sources:
        answer += f"\n\nSources:" + str(sources)
    else:
        answer += "\nNo sources found"

    await cl.Message(content=answer).send()