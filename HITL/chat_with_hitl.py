from langgraph.graph import StateGraph,START,END
from langgraph.types import interrupt,Command
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import AIMessage
from langchain_core.tools import tool
from dotenv import load_dotenv
import datetime
import requests
import os

load_dotenv()



llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="conversational",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    provider="auto",  # let Hugging Face choose the best provider for you
)

chat_model = ChatHuggingFace(llm=llm)

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
BASE_URL = "https://finnhub.io/api/v1"


@tool
def get_stock_quote(symbol:str):
    """
    Fetch real-time stock quote for a given ticker symbol.
    Returns: open, high, low, current price, previous close, % change
    """
    url = f"{BASE_URL}/quote"
    params = {
        "symbol": symbol.upper(),
        "token": FINNHUB_API_KEY
    }

    response = requests.get(url=url,params=params)
    response.raise_for_status()
    data = response.json()

    if data.get("c") == 0:
        raise ValueError(f"No data found for symbol '{symbol}'. Check the ticker.")
 
    return {
        "symbol": symbol.upper(),
        "current_price": data["c"],
        "open": data["o"],
        "high": data["h"],
        "low": data["l"],
        "previous_close": data["pc"],
        "change": round(data["c"] - data["pc"], 2),
        "change_pct": round(((data["c"] - data["pc"]) / data["pc"]) * 100, 2),
        "timestamp": datetime.date.fromtimestamp(data["t"]).strftime("%Y-%m-%d %H:%M:%S"),
    }

@tool
def purchase_stock(symbol: str, quantity: int) -> dict:
    """
    Simulate purchasing a given quantity of a stock symbol.

    HUMAN-IN-THE-LOOP:
    Before confirming the purchase, this tool will interrupt
    and wait for a human decision ("yes" / anything else).
    """
    # This pauses the graph and returns control to the caller
    decision = interrupt(f"Approve buying {quantity} shares of {symbol}? (yes/no)")

    if isinstance(decision, str) and decision.lower() == "yes":
        return {
            "status": "success",
            "message": f"Purchase order placed for {quantity} shares of {symbol}.",
            "symbol": symbol,
            "quantity": quantity,
        }
    
    else:
        return {
            "status": "cancelled",
            "message": f"Purchase of {quantity} shares of {symbol} was declined by human.",
            "symbol": symbol,
            "quantity": quantity,
        }

tools = [get_stock_quote,purchase_stock]

llm_with_tools = chat_model.bind_tools(tools)






