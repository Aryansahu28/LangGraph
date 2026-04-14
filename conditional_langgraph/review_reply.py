from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.graph import StateGraph,START,END
from pydantic import BaseModel,Field
from typing import Literal,Optional
from dotenv import load_dotenv

load_dotenv()

class ReviewState(BaseModel):
    review: str = Field(description="Review given by user")
    

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    provider="auto",  # let Hugging Face choose the best provider for you
)

chat_model = ChatHuggingFace(llm=llm)

