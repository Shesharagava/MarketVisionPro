from langchain.agents import create_agent
from langchain_community.chat_models import ChatOllama

# Change model here: "phi3", "mistral", "llama3.1"
llm = ChatOllama(model="phi3")

agent = create_agent(
    model=llm,
    tools=[],
    system_prompt="You are a finance data assistant."
)

res = agent.invoke({
    "messages":[
        {"role":"user", "content":input("Enter your question: ")}
    ]
})

print(res["messages"][1].content)
