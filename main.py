# main.py - FastAPI 메인 애플리케이션
import os
import asyncio
import traceback
from typing import List, Dict, Optional
from datetime import datetime
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# 환경 변수 로드
from dotenv import load_dotenv
load_dotenv()

# RAG 시스템 import
from rag_system import create_rag_system

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================================
# 📋 Pydantic 모델 정의
# =====================================

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    timestamp: str
    sources: Optional[List[Dict]] = None
    response_time: Optional[float] = None

class SearchRequest(BaseModel):
    query: str
    k: Optional[int] = 3
    filter: Optional[Dict] = None

class SearchResponse(BaseModel):
    query: str
    results: List[Dict]
    total_results: int
    search_time: Optional[float] = None

class HealthResponse(BaseModel):
    status: str
    message: str
    timestamp: str
    services: Dict[str, str]
    uptime: Optional[float] = None

class FilterRequest(BaseModel):
    category: Optional[str] = None
    cooking_method: Optional[str] = None
    difficulty: Optional[str] = None
    max_time: Optional[str] = None

# =====================================
# 🤖 FastAPI RAG 래퍼 클래스
# =====================================

class FastAPIRAGWrapper:
    """FastAPI용 RAG 시스템 래퍼"""
    
    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.start_time = datetime.now()
        self.request_count = 0
        self.conversation_history = {}
        logger.info("FastAPI RAG 래퍼 초기화 완료")
    
    async def chat(self, message: str, conversation_id: str = None) -> Dict:
        """챗봇 대화 처리"""
        start_time = datetime.now()
        self.request_count += 1
        
        try:
            # 대화 ID 생성
            if not conversation_id:
                conversation_id = f"conv_{int(datetime.now().timestamp())}"
            
            # RAG 시스템으로 응답 생성
            response = self.rag_system.ask(message)
            
            # 검색된 소스 정보 가져오기 (가능한 경우)
            sources = []
            try:
                if hasattr(self.rag_system, 'retriever') and self.rag_system.retriever:
                    docs = self.rag_system.retriever.get_relevant_documents(message)
                    for doc in docs[:3]:  # 상위 3개만
                        source = {
                            "title": doc.metadata.get("title", "제목없음"),
                            "category": doc.metadata.get("category", "기타"),
                            "difficulty": doc.metadata.get("difficulty", "보통"),
                            "time": doc.metadata.get("time", ""),
                            "cooking_method": doc.metadata.get("cooking_method", ""),
                            "content_preview": doc.page_content[:100] + "..."
                        }
                        sources.append(source)
            except Exception as e:
                logger.warning(f"소스 정보 가져오기 실패: {e}")
            
            # 대화 히스토리 저장
            if conversation_id not in self.conversation_history:
                self.conversation_history[conversation_id] = []
            
            self.conversation_history[conversation_id].append({
                "message": message,
                "response": response,
                "timestamp": datetime.now().isoformat(),
                "sources": sources
            })
            
            # 응답 시간 계산
            response_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "response": response,
                "conversation_id": conversation_id,
                "timestamp": datetime.now().isoformat(),
                "sources": sources,
                "response_time": response_time
            }
            
        except Exception as e:
            logger.error(f"챗봇 응답 생성 실패: {e}")
            raise HTTPException(status_code=500, detail=f"응답 생성 실패: {str(e)}")
    
    async def search(self, query: str, k: int = 3, filters: Dict = None) -> Dict:
        """레시피 검색"""
        start_time = datetime.now()
        
        try:
            # 벡터 검색 수행
            if hasattr(self.rag_system, 'vector_store') and self.rag_system.vector_store:
                # 필터 적용 검색
                if filters:
                    # Pinecone 필터 형식으로 변환
                    pinecone_filter = {}
                    for key, value in filters.items():
                        if value:
                            pinecone_filter[key] = {"$eq": value}
                    
                    docs = self.rag_system.vector_store.similarity_search(
                        query, k=k, filter=pinecone_filter if pinecone_filter else None
                    )
                else:
                    docs = self.rag_system.vector_store.similarity_search(query, k=k)
                
                # 결과 정리
                results = []
                for doc in docs:
                    result = {
                        "title": doc.metadata.get("title", "제목없음"),
                        "category": doc.metadata.get("category", "기타"),
                        "difficulty": doc.metadata.get("difficulty", "보통"),
                        "time": doc.metadata.get("time", ""),
                        "servings": doc.metadata.get("servings", ""),
                        "cooking_method": doc.metadata.get("cooking_method", ""),
                        "content": doc.page_content,
                        "view_count": doc.metadata.get("view_count", 0),
                        "image_url": doc.metadata.get("image_url", "")
                    }
                    results.append(result)
                
                search_time = (datetime.now() - start_time).total_seconds()
                
                return {
                    "query": query,
                    "results": results,
                    "total_results": len(results),
                    "search_time": search_time,
                    "filters": filters
                }
            else:
                raise HTTPException(status_code=500, detail="검색 시스템이 초기화되지 않았습니다")
                
        except Exception as e:
            logger.error(f"검색 실패: {e}")
            raise HTTPException(status_code=500, detail=f"검색 실패: {str(e)}")
    
    async def health_check(self) -> Dict:
        """시스템 상태 확인"""
        try:
            services = {}
            uptime = (datetime.now() - self.start_time).total_seconds()
            
            # RAG 시스템 상태 확인
            if self.rag_system:
                services["rag_system"] = "정상"
                
                # Pinecone 상태 확인
                if hasattr(self.rag_system, 'index') and self.rag_system.index:
                    try:
                        stats = self.rag_system.index.describe_index_stats()
                        services["pinecone"] = f"정상 ({stats.total_vector_count}개 벡터)"
                    except:
                        services["pinecone"] = "연결 오류"
                
                # LLM 상태 확인
                if hasattr(self.rag_system, 'llm') and self.rag_system.llm:
                    services["llm"] = "정상"
                
                # 벡터 스토어 상태 확인
                if hasattr(self.rag_system, 'vector_store') and self.rag_system.vector_store:
                    services["vector_store"] = "정상"
            else:
                services["rag_system"] = "초기화되지 않음"
            
            # 전체 상태 판단
            healthy_services = [s for s in services.values() if "정상" in s]
            if len(healthy_services) == len(services):
                status = "healthy"
                message = "모든 서비스가 정상 작동 중"
            elif len(healthy_services) > 0:
                status = "degraded"
                message = "일부 서비스에 문제가 있음"
            else:
                status = "unhealthy"
                message = "서비스에 심각한 문제가 있음"
            
            return {
                "status": status,
                "message": message,
                "timestamp": datetime.now().isoformat(),
                "services": services,
                "uptime": uptime,
                "request_count": self.request_count
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"상태 확인 실패: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "services": {"error": str(e)},
                "uptime": 0
            }

# =====================================
# 🚀 FastAPI 앱 생성
# =====================================

app = FastAPI(
    title="KADX Recipe RAG API",
    description="한국 농식품 레시피 기반 RAG 챗봇 API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 배포시에는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 변수
rag_wrapper = None

# =====================================
# 🎯 API 엔드포인트들
# =====================================

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 RAG 시스템 초기화"""
    global rag_wrapper
    try:
        logger.info("🚀 RAG 시스템 초기화 중...")
        rag_system = create_rag_system()
        rag_wrapper = FastAPIRAGWrapper(rag_system)
        logger.info("✅ RAG 시스템 초기화 완료!")
    except Exception as e:
        logger.error(f"❌ RAG 시스템 초기화 실패: {e}")
        # 개발 중에는 서버를 계속 실행하되, 에러 상태로 표시
        rag_wrapper = None

@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "🍳 KADX Recipe RAG API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "chat": "POST /chat",
            "search": "POST /search", 
            "health": "GET /health",
            "categories": "GET /categories",
            "stats": "GET /stats"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """시스템 상태 확인"""
    if not rag_wrapper:
        return HealthResponse(
            status="error",
            message="RAG 시스템이 초기화되지 않았습니다",
            timestamp=datetime.now().isoformat(),
            services={"rag_system": "초기화되지 않음"}
        )
    
    result = await rag_wrapper.health_check()
    return HealthResponse(**result)

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """메인 챗봇 엔드포인트"""
    if not rag_wrapper:
        raise HTTPException(status_code=503, detail="RAG 시스템이 초기화되지 않았습니다")
    
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="메시지가 비어있습니다")
    
    try:
        result = await rag_wrapper.chat(request.message, request.conversation_id)
        return ChatResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"챗봇 응답 실패: {e}")
        raise HTTPException(status_code=500, detail=f"챗봇 응답 실패: {str(e)}")

@app.post("/search", response_model=SearchResponse)
async def search_recipes(request: SearchRequest):
    """레시피 검색 엔드포인트"""
    if not rag_wrapper:
        raise HTTPException(status_code=503, detail="RAG 시스템이 초기화되지 않았습니다")
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="검색어가 비어있습니다")
    
    try:
        result = await rag_wrapper.search(request.query, request.k, request.filter)
        return SearchResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"검색 실패: {e}")
        raise HTTPException(status_code=500, detail=f"검색 실패: {str(e)}")

@app.get("/categories")
async def get_categories():
    """카테고리 목록 조회"""
    categories = {
        "categories": [
            "국물요리", "볶음요리", "찜요리", "구이요리",
            "메인반찬", "밑반찬", "나물요리", "면요리", 
            "밥요리", "디저트", "음료", "기타"
        ],
        "cooking_methods": [
            "끓이기", "볶기", "굽기", "찌기", "튀기기", 
            "삶기", "무치기", "조리기", "기타"
        ],
        "difficulties": [
            "아무나", "초급", "중급", "고급"
        ]
    }
    return categories

@app.get("/stats")
async def get_stats():
    """시스템 통계 정보"""
    if not rag_wrapper:
        raise HTTPException(status_code=503, detail="RAG 시스템이 초기화되지 않았습니다")
    
    try:
        stats = {
            "system_status": "running",
            "uptime": (datetime.now() - rag_wrapper.start_time).total_seconds(),
            "request_count": rag_wrapper.request_count,
            "conversation_count": len(rag_wrapper.conversation_history),
            "timestamp": datetime.now().isoformat()
        }
        
        # Pinecone 통계 추가
        if hasattr(rag_wrapper.rag_system, 'index') and rag_wrapper.rag_system.index:
            try:
                pinecone_stats = rag_wrapper.rag_system.index.describe_index_stats()
                stats["pinecone"] = {
                    "total_vectors": pinecone_stats.total_vector_count,
                    "index_name": rag_wrapper.rag_system.index_name
                }
            except:
                stats["pinecone"] = {"error": "통계 조회 실패"}
        
        return stats
        
    except Exception as e:
        logger.error(f"통계 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"통계 조회 실패: {str(e)}")

# =====================================
# 🔥 서버 실행 함수
# =====================================

def run_server(host: str = None, port: int = None):
    """서버 실행"""
    host = host or os.getenv("HOST", "127.0.0.1")
    port = port or int(os.getenv("PORT", 8000))
    reload = os.getenv("RELOAD", "True").lower() == "true"
    
    print("🍳 KADX Recipe RAG 서버 시작!")
    print(f"📋 API 문서: http://{host}:{port}/docs")
    print(f"🔍 상태 확인: http://{host}:{port}/health")
    print(f"💬 채팅 테스트: http://{host}:{port}/docs#/default/chat_chat_post")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

if __name__ == "__main__":
    run_server()
