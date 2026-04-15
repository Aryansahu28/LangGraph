from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.graph import StateGraph, START, END
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import Literal,Optional
from dotenv import load_dotenv

load_dotenv()


class ReviewState(BaseModel):
    review: str
    Sentiment: Optional[Literal["positive", "negative"]] = None
    diagnosis: Optional[dict] = None
    response: Optional[str] = None


class DiagnosisSchema(BaseModel):
    issue_type: Literal["UX", "Performance", "Bug", "Support", "Other"] = Field(
        description="The category of issue mentioned in the review"
    )
    tone: Literal["angry", "frustrated", "disappointed", "calm"] = Field(
        description="The emotional tone expressed by the user"
    )
    urgency: Literal["low", "medium", "high"] = Field(
        description="How urgent or critical the issue appears to be"
    )


class SentimentSchema(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="Sentiment of the review"
    )


llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    provider="auto",  # let Hugging Face choose the best provider for you
)

chat_model = ChatHuggingFace(llm=llm)


parser1 = PydanticOutputParser(pydantic_object=SentimentSchema)
parser2 = PydanticOutputParser(pydantic_object=DiagnosisSchema)


def find_sentiment(state: ReviewState):
    prompt = PromptTemplate(
        template="""
            Provide me the sentiment of the following review \n {review}\n{format_instructions}
        """,
        input_variables=["review"],
        partial_variables={"format_instructions": parser1.get_format_instructions()},
    )

    chain = prompt | chat_model | parser1

    sentiment = chain.invoke({"review": state.review})
    print(sentiment)

    return {"Sentiment": sentiment.sentiment}

def check_condition(state:ReviewState)->Literal['positive_response','run_diagnosis']:
    if state.Sentiment =='positive':
        return 'positive_response'
    else:
        return 'run_diagnosis'

def positive_response(state:ReviewState):
    prompt = f"""Write a warm thank-you message in response to this review:
    \n\n\"{state.review}\"\n
Also, kindly ask the user to leave feedback on our website."""

    result =chat_model.invoke(prompt)
    print(result)

    return {'response':result.content}

def run_diagnosis(state:ReviewState):
    prompt = PromptTemplate(
        template="""
            Diagnose this negative review:\n\n{review}\n\n{format_instructions}
    Return issue_type, tone, and urgency.
        """,
        input_variables=["review"],
        partial_variables={"format_instructions": parser2.get_format_instructions()},
    )

    chain = prompt | chat_model | parser2

    result = chain.invoke({'review':state.review})

    return {'diagnosis':result.model_dump()}

def negative_response(state: ReviewState):

    diagnosis = state.diagnosis

    prompt = f"""You are a support assistant.
The user had a '{diagnosis['issue_type']}' issue, sounded '{diagnosis['tone']}', and marked urgency as '{diagnosis['urgency']}'.
Write an empathetic, helpful resolution message.
"""
    response = chat_model.invoke(prompt).content

    return {'response': response}


graph = StateGraph(ReviewState)

graph.add_node('find_sentiment', find_sentiment)
graph.add_node('positive_response', positive_response)
graph.add_node('run_diagnosis', run_diagnosis)
graph.add_node('negative_response', negative_response)

graph.add_edge(START, 'find_sentiment')

graph.add_conditional_edges('find_sentiment', check_condition)

graph.add_edge('positive_response', END)

graph.add_edge('run_diagnosis', 'negative_response')
graph.add_edge('negative_response', END)

workflow = graph.compile()


intial_state={
    'review': "I have been trying to log in for over an hour now, and the app keeps freezing on the authentication screen. I even tried reinstalling it, but no luck. This kind of bug is unacceptable, especially when it affects basic functionality."
}
result = workflow.invoke(intial_state)

print(result)



