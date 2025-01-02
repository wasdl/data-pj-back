from fastapi import FastAPI
from pydantic import BaseModel
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import asyncio

load_dotenv()
openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class AssistantRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None



@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/assistant")
async def assistant_endpoint(req: AssistantRequest):
    try:
        assistant = await openai.beta.assistants.retrieve("asst_otYoHTbOy35yrh1toMG6he2f")
        
        if req.thread_id:
            thread_id = req.thread_id
            await openai.beta.threads.messages.create(
                thread_id=thread_id, role="user", content=req.message
            )
        else:
            thread = await openai.beta.threads.create(
                messages=[{"role": "user", "content": req.message}]
            )
            thread_id = thread.id

        run = await openai.beta.threads.runs.create(
            thread_id=thread_id, 
            assistant_id=assistant.id
        )
        
        # run이 완료될 때까지 대기
        while True:
            run_status = await openai.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run.id
            )
            if run_status.status == 'completed':
                break
            elif run_status.status == 'failed':
                return {"error": "Run failed"}
            await asyncio.sleep(1)  # 1초마다 상태 체크

        messages = await openai.beta.threads.messages.list(thread_id=thread_id)
        assistant_reply = messages.data[0].content[0].text.value

        return {"reply": assistant_reply, "thread_id": thread_id}
    
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
