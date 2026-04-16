from langgraph.graph import START,END,StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import BaseMessage,HumanMessage
from typing import Annotated,TypedDict
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
import sqlite3

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    provider="auto",  # let Hugging Face choose the best provider for you
)

chat_model = ChatHuggingFace(llm=llm)

conn = sqlite3.connect(database="chatbot.db",check_same_thread=False)



class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]

def chatbot(state:ChatState):
    message = state['messages']
    result = chat_model.invoke(message)
    return {'messages':[result]}

checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)

graph.add_node('chatbot',chatbot)

graph.add_edge(START,'chatbot')
graph.add_edge('chatbot',END)

workflow = graph.compile(checkpointer=checkpointer)


def retrieve_all_threads():
    all_threads = set()
    for check in checkpointer.list(None):
        all_threads.add(check.config['configurable']['thread_id'])
    # print(all_threads)
    return list(all_threads)

# CONFIG = {"configurable": {"thread_id": "thread_1"}}

# result = workflow.invoke({'messages':[HumanMessage(content='What is my name?')]},config=CONFIG)
# print(result['messages'][-1].content)

# for chunk in workflow.stream(
    
# ):
#     if chunk["type"]=="messages":
#             message_chunk, metadata = chunk["data"] 
#             if message_chunk.content:
#                 print(message_chunk.content,end="", flush=True)

# print(result)
