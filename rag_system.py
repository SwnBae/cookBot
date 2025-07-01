# rag_system.py (Windows 로컬 환경용)
import os
import pandas as pd
import time
from typing import List, Dict
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Pinecone 관련 import
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# LangChain 관련 import
from langchain_upstage import ChatUpstage, UpstageEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 환경 변수 로드 (getpass 제거!)
from dotenv import load_dotenv
load_dotenv()

class KADXParser:
    """KADX CSV 데이터 파싱 클래스 (Windows 최적화)"""

    def __init__(self, csv_file_path: str):
        # Windows 경로 처리 개선
        self.csv_file_path = Path(csv_file_path).resolve()
        self.df = None
        print(f"📁 CSV 파일 경로: {self.csv_file_path}")

    def load_csv(self):
        """CSV 파일 로드 (Windows 경로 처리)"""
        try:
            # Windows에서 파일 존재 여부 확인
            if not self.csv_file_path.exists():
                print(f"❌ 파일을 찾을 수 없습니다: {self.csv_file_path}")
                print(f"💡 현재 작업 디렉토리: {Path.cwd()}")
                return False
                
            self.df = pd.read_csv(self.csv_file_path, encoding='utf-8')
            print(f"✅ CSV 로드 완료: {len(self.df)}개 레시피")
            return True
        except UnicodeDecodeError:
            try:
                self.df = pd.read_csv(self.csv_file_path, encoding='cp949')
                print(f"✅ CSV 로드 완료 (cp949): {len(self.df)}개 레시피")
                return True
            except Exception as e:
                print(f"❌ CSV 로드 실패 (cp949): {e}")
                return False
        except Exception as e:
            print(f"❌ CSV 로드 실패: {e}")
            return False

    def parse_ingredients(self, ingredient_text: str) -> List[str]:
        """재료 텍스트 파싱"""
        if pd.isna(ingredient_text) or not ingredient_text:
            return []

        import re
        ingredients = re.split(r'[|,]', str(ingredient_text))
        cleaned_ingredients = []
        for ingredient in ingredients:
            ingredient = ingredient.strip()
            if ingredient and ingredient != '[재료]':
                cleaned_ingredients.append(ingredient)

        return cleaned_ingredients

    def extract_recipe_data(self, limit: int = None) -> List[Dict]:
        """KADX 데이터에서 레시피 정보 추출"""
        if self.df is None:
            return []

        recipes = []
        df_subset = self.df.head(limit) if limit else self.df

        for idx, row in df_subset.iterrows():
            try:
                recipe_data = {
                    'id': f"kadx_{row.get('RCP_SNO', idx)}",
                    'title': row.get('RCP_TTL', '제목 없음'),
                    'category': row.get('CKG_KND_ACTO_NM', '기타'),
                    'cooking_method': row.get('CKG_MTH_ACTO_NM', '기타'),
                    'difficulty': row.get('CKG_DODF_NM', '보통'),
                    'time': row.get('CKG_TIME_NM', ''),
                    'servings': row.get('CKG_INBUN_NM', ''),
                    'description': row.get('CKG_IPDC', ''),
                    'raw_ingredients': row.get('CKG_MTRL_CN', ''),
                    'ingredients': self.parse_ingredients(row.get('CKG_MTRL_CN', '')),
                    'view_count': row.get('INQ_CNT', 0),
                    'image_url': row.get('RCP_IMG_URL', '')
                }
                recipes.append(recipe_data)
            except Exception as e:
                print(f"⚠️ 레시피 {idx} 파싱 중 오류: {e}")
                continue

        print(f"📊 {len(recipes)}개 레시피 데이터 추출 완료")
        return recipes

class CookingStepsGenerator:
    """AI를 활용한 조리법 생성 클래스"""

    def __init__(self, llm):
        self.llm = llm
        self.generation_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 한국 요리 전문가입니다.
주어진 레시피 정보를 바탕으로 간결하고 실용적인 조리법을 작성해주세요.

조리법 작성 규칙:
✅ 전체 길이는 2000자 이내로 작성
✅ 재료 준비부터 완성까지 핵심만 간결하게 설명
✅ 초보자도 이해할 수 있도록 쉽게 설명
✅ 조리 팁은 1-2개만 간단히 포함
✅ 한국어로 자연스럽게 작성

응답 형식:
레시피: [요리명]
카테고리: [요리종류]
난이도: [난이도]
소요시간: [시간]
인분: [인분]

재료:
- [핵심 재료만 간결하게 나열]

조리과정:
1. [핵심 단계만 간결하게 설명]
2. [...]

조리팁: [1-2개 핵심 팁만]"""),

            ("human", """요리명: {title}
카테고리: {category}
조리방법: {cooking_method}
난이도: {difficulty}
소요시간: {time}
인분: {servings}
설명: {description}
재료: {ingredients}

위 정보를 바탕으로 간결한 조리법을 작성해주세요.""")
        ])

    def generate_cooking_steps(self, recipe_data: Dict) -> str:
        """개별 레시피의 조리법 생성"""
        try:
            ingredients_str = ", ".join(recipe_data.get('ingredients', []))
            chain = self.generation_prompt | self.llm | StrOutputParser()

            response = chain.invoke({
                'title': recipe_data.get('title', ''),
                'category': recipe_data.get('category', ''),
                'cooking_method': recipe_data.get('cooking_method', ''),
                'difficulty': recipe_data.get('difficulty', ''),
                'time': recipe_data.get('time', ''),
                'servings': recipe_data.get('servings', ''),
                'description': recipe_data.get('description', ''),
                'ingredients': ingredients_str
            })

            return response

        except Exception as e:
            print(f"❌ 조리법 생성 중 오류: {e}")
            return f"조리법 생성에 실패했습니다: {recipe_data.get('title', '')}"

class PineconeKADXRAGSystem:
    """Pinecone 기반 KADX RAG 시스템 (Windows 로컬 환경용)"""

    def __init__(self, csv_file_path: str = None, index_name: str = "kadx-recipes"):
        # 환경 변수에서 API 키 로드 (getpass 제거!)
        self.upstage_key = os.getenv("UPSTAGE_API_KEY")
        self.pinecone_key = os.getenv("PINECONE_API_KEY")
        
        if not self.upstage_key or not self.pinecone_key:
            raise ValueError("""
❌ API 키가 설정되지 않았습니다!

다음을 확인해주세요:
1. .env 파일이 프로젝트 루트에 있는지 확인
2. .env 파일에 다음 내용이 있는지 확인:
   UPSTAGE_API_KEY=your_actual_key_here
   PINECONE_API_KEY=your_actual_key_here
3. .env 파일에 실제 API 키가 입력되어 있는지 확인
""")
        
        # Windows 경로 처리 개선
        if csv_file_path:
            self.csv_file_path = str(Path(csv_file_path).resolve())
        else:
            # 환경 변수에서 경로 가져오기
            default_path = os.getenv("CSV_FILE_PATH", r".\data\241226.csv")
            self.csv_file_path = str(Path(default_path).resolve())
        
        self.index_name = index_name
        
        print(f"🔑 API 키 확인 완료")
        print(f"📁 CSV 파일 경로: {self.csv_file_path}")
        
        # Pinecone 초기화
        self.pc = Pinecone(api_key=self.pinecone_key)
        
        # LangChain 구성요소 초기화
        self.llm = ChatUpstage(model="solar-pro")
        self.embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
        
        # 파서와 생성기 초기화
        self.kadx_parser = KADXParser(self.csv_file_path)
        self.steps_generator = CookingStepsGenerator(self.llm)
        
        # RAG 구성요소
        self.vector_store = None
        self.retriever = None
        self.chain = None
        self.index = None
        
        print("🤖 Pinecone 기반 KADX RAG 시스템 초기화 완료!")

    def create_or_connect_index(self):
        """Pinecone 인덱스 생성 또는 연결"""
        try:
            # 기존 인덱스 목록 확인
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.index_name in existing_indexes:
                print(f"✅ 기존 인덱스 '{self.index_name}' 발견! 연결합니다.")
                self.index = self.pc.Index(self.index_name)
                
                # 인덱스 정보 확인
                stats = self.index.describe_index_stats()
                vector_count = stats.total_vector_count
                print(f"📊 기존 벡터 수: {vector_count}개")
                
                if vector_count > 0:
                    # 기존 벡터 스토어 연결
                    self.vector_store = PineconeVectorStore(
                        index=self.index,
                        embedding=self.embeddings,
                        text_key="text"
                    )
                    return True
                else:
                    print("📝 인덱스가 비어있습니다. 새로 구축합니다.")
                    return False
            else:
                print(f"🔨 새 인덱스 '{self.index_name}' 생성 중...")
                
                # 새 인덱스 생성
                self.pc.create_index(
                    name=self.index_name,
                    dimension=4096,  # Upstage Solar Embedding 차원
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
                
                # 인덱스 생성 대기
                print("⏳ 인덱스 생성 대기 중...")
                time.sleep(10)
                
                self.index = self.pc.Index(self.index_name)
                print(f"✅ 새 인덱스 '{self.index_name}' 생성 완료!")
                return False
                
        except Exception as e:
            print(f"❌ 인덱스 생성/연결 실패: {e}")
            return False

    def setup_retriever(self):
        """리트리버 설정"""
        if not self.vector_store:
            print("❌ 벡터 스토어가 설정되지 않았습니다")
            return False

        try:
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )
            print("✅ 리트리버 설정 완료!")
            return True
        except Exception as e:
            print(f"❌ 리트리버 설정 실패: {e}")
            return False

    def create_rag_chain(self):
        """RAG 체인 생성"""
        print("🔗 RAG 체인 생성 중...")

        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 한국 요리 전문가이자 친근한 요리 도우미입니다.
사용자의 질문에 대해 제공된 레시피 정보를 바탕으로 정확하고 도움이 되는 답변을 해주세요.

답변 스타일:
✅ 존댓말로 친근하게 답변
✅ 재료와 조리과정을 명확히 구분하여 설명
✅ 조리팁과 대안 재료 제안
✅ 난이도와 소요시간 명시
✅ 질문이 모호하면 관련 레시피들 추천

제공된 레시피 정보:
{context}"""),

            ("human", "{question}")
        ])

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        self.chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | rag_prompt
            | self.llm
            | StrOutputParser()
        )

        print("✅ RAG 체인 생성 완료!")
        return True

    def setup_system(self, recipe_limit: int = 20, force_rebuild: bool = False):
        """전체 시스템 설정 (로컬 환경용 간소화)"""
        print("🚀 Pinecone 기반 KADX RAG 시스템 설정 시작!")
        print("=" * 60)

        # 1. Pinecone 인덱스 생성/연결
        has_existing_data = self.create_or_connect_index()

        # 2. 기존 데이터가 있으면 사용 (로컬에서는 재구축 생략)
        if has_existing_data:
            print("⚡ 기존 Pinecone 데이터를 사용합니다!")
            
            # 리트리버 및 체인 설정
            if self.setup_retriever() and self.create_rag_chain():
                print("✅ 기존 데이터로 RAG 시스템 설정 완료!")
                print("=" * 60)
                return self
            else:
                print("❌ 기존 데이터 연결 실패")
                return self
        else:
            print("❌ 기존 데이터가 없습니다. 코랩에서 먼저 데이터를 구축해주세요.")
            print("💡 코랩에서 다음 명령어로 데이터 구축:")
            print("   rag = setup_pinecone_kadx_rag(csv_path, recipe_limit=1000, force_rebuild=True)")
            return self

    def ask(self, question: str) -> str:
        """레시피 질문하기"""
        if not self.chain:
            return "❌ RAG 시스템이 초기화되지 않았습니다."

        try:
            response = self.chain.invoke(question)
            return response
        except Exception as e:
            print(f"❌ 질문 처리 중 오류 발생: {e}")
            return f"❌ 오류 발생: {e}"

    def get_index_info(self):
        """Pinecone 인덱스 정보 조회"""
        if not hasattr(self, 'index') or not self.index:
            return "❌ Pinecone 인덱스가 연결되지 않았습니다."
        
        try:
            stats = self.index.describe_index_stats()
            
            info = f"📊 Pinecone 인덱스 정보:\n"
            info += f"   • 인덱스 이름: {self.index_name}\n"
            info += f"   • 총 벡터 수: {stats.total_vector_count}개\n"
            info += f"   • 차원 수: 4096\n"
            info += f"   • 메트릭: cosine\n"
            
            return info
        except Exception as e:
            return f"❌ 인덱스 정보 조회 실패: {e}"

# 간소화된 생성 함수
def create_rag_system():
    """RAG 시스템 생성 (환경 변수 기반)"""
    try:
        print("🔧 RAG 시스템 생성 중...")
        
        # CSV 파일 경로 (환경 변수 또는 기본값)
        csv_path = os.getenv("CSV_FILE_PATH")
        if not csv_path:
            # 기본 경로들 시도
            possible_paths = [
                r".\data\241226.csv",
                r"data\241226.csv", 
                r"241226.csv"
            ]
            
            for path in possible_paths:
                if Path(path).exists():
                    csv_path = path
                    break
            
            if not csv_path:
                print("❌ CSV 파일을 찾을 수 없습니다!")
                print("💡 다음 중 하나를 확인해주세요:")
                print("   1. data/241226.csv 파일이 있는지 확인")
                print("   2. .env 파일에 CSV_FILE_PATH 설정")
                raise FileNotFoundError("CSV 파일을 찾을 수 없습니다")
        
        rag_system = PineconeKADXRAGSystem(csv_path)
        rag_system.setup_system(recipe_limit=1000, force_rebuild=False)
        
        print("✅ RAG 시스템 생성 완료!")
        return rag_system
        
    except Exception as e:
        print(f"❌ RAG 시스템 생성 실패: {e}")
        raise e

# 테스트 함수
def test_rag_system():
    """RAG 시스템 테스트"""
    try:
        rag = create_rag_system()
        
        print("\n🧪 RAG 시스템 테스트")
        print("=" * 40)
        
        # 인덱스 정보 확인
        print(rag.get_index_info())
        
        # 테스트 질문
        test_question = "김치찌개 만드는 법 알려줘"
        print(f"\n❓ 테스트 질문: {test_question}")
        
        answer = rag.ask(test_question)
        print(f"🤖 답변: {answer[:200]}...")
        
        return rag
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        return None

if __name__ == "__main__":
    # 직접 실행 시 테스트
    print("🍳 KADX Recipe RAG 시스템 (Windows 로컬)")
    test_rag_system()
