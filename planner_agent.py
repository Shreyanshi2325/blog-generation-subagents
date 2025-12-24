from ast import Import
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import time
from groq import RateLimitError

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
    model="llama-3.1-8b-instant",
    temperature=0.1
)

planner_agent = create_agent(
    model=llm,
    system_prompt="""
You are a planning specialist.
Given a topic, break it into:
- Blog goals
- Target audience
- Content structure
Return a clear plan.
"""
)

research_agent = create_agent(
    model=llm,
    system_prompt="""
You are a research specialist.
Gather:
- Product details
- Sustainability benefits
- SEO keywords
Return concise bullet points.
"""
)


# writer_validator_agent = create_agent(
#     model=llm,
#     system_prompt="""
# You are a writing and validation specialist.

# Steps:
# 1. Write a complete blog post
# 2. Validate tone, clarity, and completeness
# 3. Fix issues if any
# Return final polished blog.
# """
# )

writer_validator_agent = create_agent(
    model=llm,
    system_prompt="""
You are a writing and validation specialist.

You MUST follow this output format exactly:

=== BLOG ===
<final polished blog content>

=== VALIDATION ===
- Tone: OK / Needs Fix
- Clarity: OK / Needs Fix
- Completeness: OK / Needs Fix

Rules:
- Do NOT explain what you did
- Do NOT summarize the task
- Output ONLY the sections above
"""
)



# social_agent = create_agent(
#     model=llm,
#     system_prompt="""
# You are a social media specialist.
# Create a suitable caption and keep them engaging and on-brand.
# """
# )

social_agent = create_agent(
    model=llm,
    system_prompt="""
You are a social media specialist.

Output format:

Instagram:
<caption>

LinkedIn:
<caption>

Twitter/X:
<caption>
"""
)



SUBAGENTS = {
    "planner": planner_agent,
    "research": research_agent,
    "writer_validator": writer_validator_agent,
    "social": social_agent,
}

# @tool
# def task(agent_name: str, description: str) -> str:
#     """
#     Launch a sub-agent for a task.

#     Available agents:
#     - planner: content planning
#     - research: research and SEO
#     - writer_validator: blog writing + validation
#     - social: social media captions
#     """
#     agent = SUBAGENTS[agent_name]

#     result = agent.invoke({
#         "messages": [
#             {"role": "user", "content": description}
#         ]
#     })

#     return result["messages"][-1].content

@tool
def task(agent_name: str, description: str) -> str:
    """
    Launch a sub-agent for a task.

    Available agents:
    - planner: content planning
    - research: research and SEO
    - writer_validator: blog writing + validation
    - social: social media captions
    """
    agent = SUBAGENTS[agent_name]

    result = safe_invoke(
        agent,
        [{"role": "user", "content": description}]
    )

    return result["messages"][-1].content



# supervisor_agent = create_agent(
#     model=llm,
#     tools=[task],
#     system_prompt="""
# You are a supervisor marketing agent.

# Given a topic, you must:
# 1. Call planner agent to plan content
# 2. Call research agent for facts and SEO
# 3. Call writer_validator agent to write and validate a blog
# 4. Call social agent to generate captions
# 5. Combine everything into a final structured response

# You MUST use the task tool for delegation.
# """
# )


supervisor_agent = create_agent(
    model=llm,
    tools=[task],
    system_prompt="""
You are a supervisor marketing agent.

Workflow:
1. Use planner agent → get plan
2. Use research agent → get facts + SEO
3. Use writer_validator agent → get BLOG and VALIDATION
4. Use social agent → get captions

FINAL OUTPUT FORMAT:

=== CONTENT PLAN ===
<planner output>

=== RESEARCH ===
<research output>

=== BLOG ===
<blog content>

=== SOCIAL CAPTIONS ===
<captions>

Rules:
- Do NOT explain the process
- Do NOT add meta commentary
- Present final content only
"""
)

query = "Eco-friendly Bamboo Toothbrush"

response = supervisor_agent.invoke({
    "messages": [
        {"role": "user", "content": query}
    ]
})

print(response["messages"][-1].content)
