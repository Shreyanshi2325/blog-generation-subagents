import time
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import time
from groq import RateLimitError
load_dotenv()
load_dotenv()

# Enable LangSmith tracing - NO UI SETUP REQUIRED
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGCHAIN_TRACING_V2"] = "true" 
os.environ["LANGCHAIN_PROJECT"] = "blog-generation-subagents"  # Your exact project name

def safe_invoke(agent, messages, retries=3):
    for i in range(retries):
        try:
            return agent.invoke({"messages": messages})
        except RateLimitError as e:
            if i == retries - 1:
                raise
            time.sleep(0.5)  # backoff


load_dotenv()

assert os.getenv("GROQ_API_KEY"), "GROQ_API_KEY not found in environment"


llm = ChatGroq(
    model="openai/gpt-oss-120b",   # must exist in `ollama list`
    temperature=0.1
)


title_agent = create_agent(
    model=llm,
    system_prompt="""
You are an expert blog content writer. Generate a blog title. This title should be creative and SEO friendly.
"""
)

content_agent = create_agent(
    model=llm,
    system_prompt="""
You are expert blog writer. Generate a detailed blog content with detailed breakdown such 
"""
)


SUBAGENTS = {
    "title": title_agent,
    "content": content_agent
}


@tool
def task(agent_name: str, description: str) -> str:
    """
    Launch a sub-agent for a task.

    Available agents:
    - title
    - content
    """
    agent = SUBAGENTS[agent_name]

    result = agent.invoke({
        "messages": [
            {"role": "user", "content": description}
        ]
    })

    return result["messages"][-1].content


supervisor_agent = create_agent(
    model=llm,
    tools=[task],
    system_prompt="""
You are a supervisor marketing agent.
From the user query, you can invoke the below tools as per requirement.
-title agent:Use this agent to generate a catchy title
-content agent: Use this agent to generate appropriate content within word limit of 100-200 words only

FINAL OUTPUT FORMAT:

CAPTION
<captions>

BLOG
<blog content>

Rules:
- Do NOT explain the process
- Do NOT add meta commentary
- Present final content only
-Remove uncecessary special characters/unicodes from final response
"""
)

# =========================
# Run the System
# =========================
if __name__ == "__main__":
    query = input("Enter blog topic: ").strip()

    if not query:
        raise ValueError("Query cannot be empty")

    response = supervisor_agent.invoke({
        "messages": [
            {"role": "user", "content": query}
        ]
    })

    print("\n" + "="*50 + "\n")
    print(response["messages"][-1].content)

