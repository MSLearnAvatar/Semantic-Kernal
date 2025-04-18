import asyncio
import os
import time
from typing import Dict, Optional

import uvicorn

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from azure.identity import ClientSecretCredential
from azure.ai.projects.aio import AIProjectClient
from semantic_kernel.agents import AzureAIAgent
import multiprocessing


# 환경변수 로드
load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


origins = ["Your Azure static wep app"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

max_requests = 1000
max_requests_jitter = 50

log_file = "-"

bind = "0.0.0.0:3100"

worker_class = "uvicorn.workers.UvicornWorker"
workers = (multiprocessing.cpu_count() * 2) + 1


# 환경변수 설정
AZURE_TENANT_ID = os.getenv("AZURE_TENANT_ID")
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
SPEECH_KEY = os.getenv("SPEECH_KEY")
SPEECH_REGION = os.getenv("SPEECH_REGION")
RESOURCE_GROUP = os.getenv("RESOURCE_GROUP")
PROJECT_NAME = os.getenv("PROJECT_NAME")
MODEL_DEPLOYMENT_NAME = os.getenv("MODEL_DEPLOYMENT_NAME")
PROJECT_CONN_STR = "PROJECT_CONN_STR"

# 에이전트 ID 정의
AGENT_IDS = {
    "search": "Search agent ID",
    "verification": "Verification agent ID",
    "talk": "Talk Agent ID",
    "text": "Text Agent ID"
}

class AgentManager:
    def __init__(self, client):
        self.client = client
        self.agents = {}
        self.threads = []
        self.agent_definitions_cache = {}  # 에이전트 정의 캐싱을 위한 딕셔너리

    async def initialize_agents(self):
        """
        모든 에이전트를 병렬로 초기화하고 캐싱을 활용합니다.
        """
        start_time = time.time()
        print("에이전트 초기화 시작...")
        
        # 병렬로 모든 에이전트 초기화
        initialization_tasks = []
        for role, agent_id in AGENT_IDS.items():
            initialization_tasks.append(self._initialize_agent(role, agent_id))
        
        # 모든 초기화 태스크를 동시에 실행
        await asyncio.gather(*initialization_tasks)
        
        end_time = time.time()
        print(f"✅ 모든 에이전트 초기화 완료 (소요 시간: {end_time - start_time:.2f}초)")

    async def _initialize_agent(self, role: str, agent_id: str):
        """
        단일 에이전트를 초기화하고 캐싱을 활용합니다.
        """
        # 캐시에서 에이전트 정의 확인
        if agent_id in self.agent_definitions_cache:
            print(f"🔄 {role.capitalize()} 에이전트 정의를 캐시에서 로드합니다.")
            agent_def = self.agent_definitions_cache[agent_id]
        else:
            # 캐시에 없으면 API에서 가져오기
            print(f"🔍 {role.capitalize()} 에이전트 정의를 API에서 가져옵니다.")
            agent_def = await self.client.agents.get_agent(agent_id=agent_id)
            # 캐시에 저장
            self.agent_definitions_cache[agent_id] = agent_def
            print(f"💾 {role.capitalize()} 에이전트 정의를 캐시에 저장했습니다.")
        
        # 에이전트 인스턴스 생성
        self.agents[role] = AzureAIAgent(client=self.client, definition=agent_def)
        print(f"✅ {role.capitalize()} 에이전트 초기화 완료.")

    async def create_thread(self):
        thread = await self.client.agents.create_thread()
        self.threads.append(thread)
        return thread

    async def cleanup(self):
        cleanup_tasks = []
        for thread in self.threads:
            cleanup_tasks.append(self._cleanup_thread(thread))
        
        # 병렬로 모든 스레드 정리
        await asyncio.gather(*cleanup_tasks)
    
    async def _cleanup_thread(self, thread):
        try:
            await self.client.agents.delete_thread(thread.id)
            print(f"✅ Thread {thread.id} deleted successfully.")
        except Exception as e:
            print(f"❌ Error deleting thread {thread.id}: {e}")

class VerificationPipeline:
    def __init__(self, manager):
        self.manager = manager
        self.max_attempts = 4

    async def run(self, user_query):
        attempts = 0
        response_doc = None
        
        while attempts < self.max_attempts:
            attempts += 1
            
            if response_doc is None:
                thread = await self.manager.create_thread()
                response_doc = await self.manager.agents["search"].get_response(
                    thread_id=thread.id,
                    messages=user_query
                )

            print("\n[Search Agent Response]")
            print(response_doc)
            
            thread_v = await self.manager.create_thread()
            response_content = response_doc if isinstance(response_doc, str) else str(response_doc)
            verification_response = await self.manager.agents["verification"].get_response(
                thread_id=thread_v.id,
                messages=response_content
            )

            verification_success = any(phrase in str(verification_response)
                                    for phrase in ["검증 완료", "검증완료"])
            
            if verification_success:
                print("\n✅ Verification successful! Proceeding to next agents.")
                return response_doc
                
            print(f"\n❌ Verification failed (Attempt {attempts}/{self.max_attempts}). Sending feedback to Search Agent.")
            
            thread_retry = await self.manager.create_thread()
            feedback_message = f"원래 질문: {user_query}\n\n검증 피드백: {verification_response}을 바탕으로 위의 피드백 사항을 참고하여 재검색 및 추가적인 데이터 검토를 수행해 주세요 그리고 상기된 부분을 보완하여 조정해 주시기 바랍니다"
            response_doc = await self.manager.agents["search"].get_response(
                thread_id=thread_retry.id,
                messages=feedback_message
            )
        
        print("\n❌ 검증을 하지 못하였습니다. 부적절한 입력이거나 시스템의 문제일 수 있습니다. 다시 시도해 주세요.")
        return None

class TalkAgentInteraction:
    def __init__(self, manager):
        self.manager = manager

    async def run(self, content):
        thread = await self.manager.create_thread()
        await self.manager.client.agents.create_message(
            thread_id=thread.id,
            role="user",
            content=f"{content}\n\n모든 내용을 한 번에 상세하게 제공해 주세요."
        )

        run = await self.manager.client.agents.create_run(
            thread_id=thread.id,
            agent_id=self.manager.agents["talk"].id
        )

        while True:
            run_status = await self.manager.client.agents.get_run(
                thread_id=thread.id,
                run_id=run.id
            )
            
            if run_status.status not in ["queued", "in_progress"]:
                break
                
            await asyncio.sleep(0.5)
            
        messages = await self.manager.client.agents.list_messages(thread_id=thread.id)
        response = messages.data[0].content[0].text.value if messages.data else "응답 없음"
        
        print("\n[Talk Agent Response]")
        print(response)
        
        return response

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "speech_key": SPEECH_KEY,
        "speech_region": SPEECH_REGION
    })


@app.get("/health")
def health():
    return {"status": "ok"}


@app.websocket("/api/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket 클라이언트 연결됨")

    credential = ClientSecretCredential(
        tenant_id=AZURE_TENANT_ID,
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET
    )

    try:
        project_client = AIProjectClient.from_connection_string(
            credential=credential,
            conn_str=PROJECT_CONN_STR,
            logging_enable=True
        )

        print("✅ Project client initialized.")

        async with AzureAIAgent.create_client(
            credential=credential,
            conn_str=PROJECT_CONN_STR,
            model_deployment_name=MODEL_DEPLOYMENT_NAME
        ) as client:
            manager = AgentManager(client)
            await manager.initialize_agents()
            await websocket.send_json({"status": "connected"})

            while True:
                try:
                    user_query = await websocket.receive_text()
                    print(f"사용자 쿼리 수신: {user_query}")
                    
                    # 병렬로 검색 및 검증 에이전트 실행
                    start_time = time.time()
                    
                    # 검색 에이전트 실행
                    thread_search = await manager.create_thread()
                    search_response = await manager.agents["search"].get_response(
                        thread_id=thread_search.id,
                        messages=user_query
                    )
                    
                    # 검증 에이전트 실행
                    thread_verify = await manager.create_thread()
                    verification_response = await manager.agents["verification"].get_response(
                        thread_id=thread_verify.id,
                        messages=str(search_response)
                    )
                    
                    # Talk 및 Text 에이전트 병렬 실행
                    talk_interaction = TalkAgentInteraction(manager)
                    text_response_thread = await manager.create_thread()
                    
                    # 병렬로 Talk 및 Text 에이전트 실행
                    talk_task = talk_interaction.run(search_response)
                    text_task = manager.agents["text"].get_response(
                        thread_id=text_response_thread.id,
                        messages=f"{str(search_response)}\n\n전체 내용을 한 번에 제공해주세요."
                    )
                    
                    # 두 태스크의 결과를 동시에 기다림
                    talk_response, text_response = await asyncio.gather(talk_task, text_task)
                    
                    end_time = time.time()
                    print(f"✅ 모든 에이전트 응답 완료 (소요 시간: {end_time - start_time:.2f}초)")
                    
                    # 결과 전송 - 모든 응답을 문자열로 변환하여 JSON 직렬화 가능하게 함
                    await websocket.send_json({
                        "search_response": str(search_response),
                        "verification_response": str(verification_response),
                        "talk_response": str(talk_response),
                        "text_response": str(text_response),
                    })
                    
                except Exception as e:
                    print(f"쿼리 처리 중 오류 발생: {str(e)}")
                    await websocket.send_json({"error": str(e)})
                    
    except WebSocketDisconnect:
        print("WebSocket 클라이언트 연결 종료")
    except Exception as e:
        print(f"WebSocket 처리 중 오류: {str(e)}")
        try:
            await websocket.send_json({"error": f"서버 오류: {str(e)}"})
        except:
            print("클라이언트에 오류 메시지를 보내는 중 추가 오류 발생")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
