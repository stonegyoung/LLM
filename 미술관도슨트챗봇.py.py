from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from typing import Optional

from langchain.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path='C:/mtoc/gpt.env')

# 모델
model = ChatOpenAI(
    model_name = 'gpt-4o-mini',
    temperature = 0.3
)

# 프롬프트
system_prompt = PromptTemplate(
    input_variables=["author","title"], # 들어오는 값
    template="당신은 전시회의 도슨트입니다. 미술과 연관되지 않은 내용은 대답하지 말아주세요."
)

# 메모리
memory = ConversationBufferMemory(memory_key='history', return_messges=True)

app = FastAPI()

class Art(BaseModel):
    title: str
    author: str
    chat: Optional[str] = None
    

# post로 해야 null값 none값으로 처리 됨
@app.post('/chatbot')
def chatbot(art:Art):
    prompt_result = system_prompt.format(author=art.author, title=art.title)

    custom_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(prompt_result),
        HumanMessagePromptTemplate.from_template('{history}'),
        HumanMessagePromptTemplate.from_template('{input}')
    ])
    
    # 체인
    gpt_chain = ConversationChain(
        llm = model,
        prompt = custom_prompt,
        memory = memory
    )
    
    if art.chat == None:                
        explain = f'{art.author}의 {art.title} 작품에 대한 설명을 해주세요.'        
        result = gpt_chain.invoke(explain)
        return {"explain":result['response']}
    
    else:
        result = gpt_chain.invoke(art.chat)
        return {"response": result['response']}

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=7777)
    