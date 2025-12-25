from fastapi import FastAPI
from pydantic import BaseModel
from agents import supervisor_agent

app = FastAPI(title="Blog Generator API")

class BlogRequest(BaseModel):
    topic: str

class BlogResponse(BaseModel):
    topic: str
    output: str

@app.post("/generate-blog", response_model=BlogResponse)
def generate_blog(req: BlogRequest):
    response = supervisor_agent.invoke({
        "messages": [
            {"role": "user", "content": req.topic}
        ]
    })

    return {
        "topic": req.topic,
        "output": response["messages"][-1].content
    }
