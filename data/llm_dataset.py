import os
import sys
import re
import shutil
import torch
from typing import Optional, Dict, List, Any
from torch.utils.data import Dataset

from transformers import AutoModelForCausalLM, AutoTokenizer

# 프로젝트 루트를 sys.path에 추가 (data 디렉토리에서 실행할 때를 위해)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# tokenizers 병렬 처리 경고 방지 (프로세스 fork 전에 설정해야 함)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from data.dataset import VibrationDataset
from data.feature_extract import LLM_Dataset as FeatureExtractLLMDataset

# LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

class SemanticTextSplitter:
    """
    문서 구조 기반 semantic text splitter
    Chapter, Section, Subsection 단위로 문서를 분할
    """
    
    def __init__(self, chunk_size=1000, chunk_overlap=200, min_chunk_size=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
        # 문서 구조 패턴 정의
        self.patterns = {
            'chapter': re.compile(r'^Chapter\s+(\d+(?:\.\d+)?)\s+(.+)$', re.MULTILINE | re.IGNORECASE),
            'section': re.compile(r'^(\d+\.\d+)\s+(.+)$', re.MULTILINE),
            'subsection': re.compile(r'^(\d+\.\d+\.\d+)\s+(.+)$', re.MULTILINE),
            'table': re.compile(r'^Table\s+(\d+(?:\.\d+)?)\s+(.+)$', re.MULTILINE | re.IGNORECASE),
            'figure': re.compile(r'^Figure\s+(\d+(?:\.\d+)?)\s+(.+)$', re.MULTILINE | re.IGNORECASE),
        }
    
    def _parse_structure(self, text):
        """문서 구조 파싱 - Chapter, Section, Subsection 위치 찾기"""
        structure = []
        lines = text.split('\n')
        
        current_chapter = None
        current_section = None
        current_subsection = None
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            # Chapter 검사
            chapter_match = self.patterns['chapter'].match(line_stripped)
            if chapter_match:
                current_chapter = {
                    'type': 'chapter',
                    'number': chapter_match.group(1),
                    'title': chapter_match.group(2).strip(),
                    'line': i,
                    'content_start': i
                }
                current_section = None
                current_subsection = None
                structure.append(current_chapter)
                continue
            
            # Section 검사
            section_match = self.patterns['section'].match(line_stripped)
            if section_match:
                current_section = {
                    'type': 'section',
                    'number': section_match.group(1),
                    'title': section_match.group(2).strip(),
                    'line': i,
                    'content_start': i,
                    'chapter': current_chapter['number'] if current_chapter else None
                }
                current_subsection = None
                structure.append(current_section)
                continue
            
            # Subsection 검사
            subsection_match = self.patterns['subsection'].match(line_stripped)
            if subsection_match:
                current_subsection = {
                    'type': 'subsection',
                    'number': subsection_match.group(1),
                    'title': subsection_match.group(2).strip(),
                    'line': i,
                    'content_start': i,
                    'chapter': current_chapter['number'] if current_chapter else None,
                    'section': current_section['number'] if current_section else None
                }
                structure.append(current_subsection)
                continue
            
            # Table 검사
            table_match = self.patterns['table'].match(line_stripped)
            if table_match:
                structure.append({
                    'type': 'table',
                    'number': table_match.group(1),
                    'title': table_match.group(2).strip(),
                    'line': i,
                    'content_start': i,
                    'chapter': current_chapter['number'] if current_chapter else None,
                    'section': current_section['number'] if current_section else None
                })
                continue
            
            # Figure 검사
            figure_match = self.patterns['figure'].match(line_stripped)
            if figure_match:
                structure.append({
                    'type': 'figure',
                    'number': figure_match.group(1),
                    'title': figure_match.group(2).strip(),
                    'line': i,
                    'content_start': i,
                    'chapter': current_chapter['number'] if current_chapter else None,
                    'section': current_section['number'] if current_section else None
                })
                continue
        
        return structure
    
    def split_documents(self, documents):
        """Document 리스트를 구조 기반으로 분할"""
        all_chunks = []
        
        for doc_idx, doc in enumerate(documents):
            text = doc.page_content
            source = doc.metadata.get('source', 'unknown')
            
            # 문서 구조 파싱
            structure = self._parse_structure(text)
            lines = text.split('\n')
            
            if not structure:
                # 구조가 없으면 RecursiveCharacterTextSplitter 사용
                fallback_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap
                )
                chunks = fallback_splitter.split_documents([doc])
                for chunk_idx, chunk in enumerate(chunks):
                    chunk.metadata.update({
                        'chunk_index': chunk_idx,
                        'chapter': None,
                        'section': None,
                        'content_type': 'text',
                        'char_start': text.find(chunk.page_content),
                        'char_end': text.find(chunk.page_content) + len(chunk.page_content)
                    })
                all_chunks.extend(chunks)
                continue
            
            # 구조 기반 분할
            for struct_idx, struct in enumerate(structure):
                # 현재 구조 요소의 시작 라인
                start_line = struct['line']
                
                # 다음 구조 요소의 시작 라인 (또는 문서 끝)
                if struct_idx + 1 < len(structure):
                    end_line = structure[struct_idx + 1]['line']
                else:
                    end_line = len(lines)
                
                # 해당 구간의 텍스트 추출
                section_lines = lines[start_line:end_line]
                section_text = '\n'.join(section_lines).strip()
                
                if len(section_text) < self.min_chunk_size:
                    continue
                
                # 청크가 너무 크면 RecursiveCharacterTextSplitter로 재분할
                if len(section_text) > self.chunk_size:
                    temp_doc = Document(
                        page_content=section_text,
                        metadata={'source': source}
                    )
                    fallback_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=self.chunk_size,
                        chunk_overlap=self.chunk_overlap
                    )
                    sub_chunks = fallback_splitter.split_documents([temp_doc])
                    
                    for sub_idx, sub_chunk in enumerate(sub_chunks):
                        char_start = text.find(sub_chunk.page_content)
                        sub_chunk.metadata.update({
                            'chunk_index': len(all_chunks) + sub_idx,
                            'chapter': struct.get('chapter'),
                            'section': struct.get('section'),
                            'subsection': struct.get('number') if struct['type'] == 'subsection' else None,
                            'content_type': struct['type'],
                            'struct_number': struct.get('number'),
                            'struct_title': struct.get('title'),
                            'char_start': char_start if char_start >= 0 else 0,
                            'char_end': char_start + len(sub_chunk.page_content) if char_start >= 0 else len(sub_chunk.page_content)
                        })
                    all_chunks.extend(sub_chunks)
                else:
                    # 청크 크기가 적절하면 그대로 사용
                    char_start = text.find(section_text)
                    chunk = Document(
                        page_content=section_text,
                        metadata={
                            'source': source,
                            'chunk_index': len(all_chunks),
                            'chapter': struct.get('chapter'),
                            'section': struct.get('section'),
                            'subsection': struct.get('number') if struct['type'] == 'subsection' else None,
                            'content_type': struct['type'],
                            'struct_number': struct.get('number'),
                            'struct_title': struct.get('title'),
                            'char_start': char_start if char_start >= 0 else 0,
                            'char_end': char_start + len(section_text) if char_start >= 0 else len(section_text)
                        }
                    )
                    all_chunks.append(chunk)
        
        return all_chunks

def retrieve_documents(
    retriever,
    current_knowledge: Dict[str, float],
    target_labels: str = "normal(healthy), misalignment, looseness, unbalance, bearing fault"
) -> List[Document]:
    """
    큰 변화율과 비정상 지표에 집중한 검색 쿼리 생성 및 문서 검색
    
    Args:
        retriever: 벡터 검색기
        current_knowledge: 정상 상태 대비 변화율(%) 딕셔너리
            - 형식: {'rms_x': 3622.7905, 'kurtosis_x': -71.4348, ...}
            - 양수: 정상보다 증가 (예: 316.26% = 정상 대비 316% 증가)
            - 음수: 정상보다 감소 (예: -75.28% = 정상 대비 75% 감소)
            - 변화율 = (현재값 - 정상값) / 정상값 * 100
            - 0이 아닌 값만 포함됨 (feature_extract.py에서 필터링됨)
        target_labels: 진단 대상 레이블 문자열
    
    Returns:
        검색된 Document 객체 리스트
    """
    def classify_changes(change_rates, threshold_extreme=50.0, threshold_large=20.0, threshold_moderate=10.0):
        """변화율을 극단/큰/중간으로 분류하고 절댓값 기준 내림차순 정렬"""
        sorted_changes = sorted(
            change_rates.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        extreme_features = []  # 극단적 변화 (>=50%)
        large_features = []     # 큰 변화 (20-50%)
        moderate_features = []  # 중간 변화 (10-20%)
        
        for feature, change_rate in sorted_changes:
            abs_rate = abs(change_rate)
            if abs_rate >= threshold_extreme:
                extreme_features.append((feature, change_rate))
            elif abs_rate >= threshold_large:
                large_features.append((feature, change_rate))
            elif abs_rate >= threshold_moderate:
                moderate_features.append((feature, change_rate))
        
        return extreme_features, large_features, moderate_features
    
    # 변화율 직접 사용 (이미 딕셔너리 형식이고 필터링됨)
    change_rates = current_knowledge if isinstance(current_knowledge, dict) else {}
    
    # 변화율 분류
    extreme_features, large_features, moderate_features = classify_changes(change_rates)
    
    # 고장 유형별 중요 특징 매핑 테이블
    fault_critical_features = {
        "misalignment": {
            "primary": ["order_x_2x", "order_y_2x", "order_x_1x", "order_y_1x"],
            "secondary": ["rms_x", "rms_y", "peak_freq_x", "peak_freq_y", "peak2peak_x", "peak2peak_y"]
        },
        "unbalance": {
            "primary": ["order_x_1x", "order_y_1x", "rms_x", "rms_y"],
            "secondary": ["peak_freq_x", "peak_freq_y", "peak2peak_x", "peak2peak_y", "peak_abs_x", "peak_abs_y"]
        },
        "looseness": {
            "primary": ["kurtosis_x", "kurtosis_y", "crest_factor_x", "crest_factor_y", 
                       "order_x_3x", "order_y_3x"],
            "secondary": ["skewness_x", "skewness_y", "peak2peak_x", "peak2peak_y", "var_x", "var_y"]
        },
        "bearing fault": {
            "primary": ["order_x_2x", "order_x_3x", "order_y_2x", "order_y_3x",
                       "peak_freq_x", "peak_freq_y"],
            "secondary": ["rms_freq_x", "rms_freq_y", "center_freq_x", "center_freq_y", 
                         "bpfo_peak_x", "bpfi_peak_x"]
        }
    }
    
    # 가중치 기반 고장 유형 추론
    fault_scores = {}
    all_abnormal_features = extreme_features + large_features
    
    for fault_type, features in fault_critical_features.items():
        score = 0.0
        # Primary 특징에 더 높은 가중치 (2.0)
        for feat_name, change_rate in all_abnormal_features:
            if feat_name in features["primary"]:
                score += abs(change_rate) * 2.0
            elif feat_name in features["secondary"]:
                score += abs(change_rate) * 1.0
        
        if score > 0:
            fault_scores[fault_type] = score
    
    # 점수가 높은 상위 2개 고장 유형 선택
    suspected_faults = []
    if fault_scores:
        sorted_faults = sorted(fault_scores.items(), key=lambda x: x[1], reverse=True)
        suspected_faults = [fault for fault, score in sorted_faults[:2]]
    
    # 고장 유형별 키워드 매핑
    fault_keywords = {
        "misalignment": ["2x harmonic", "second harmonic", "misalignment", "axial", "radial"],
        "unbalance": ["1x harmonic", "first harmonic", "unbalance", "rotational"],
        "looseness": ["3x harmonic", "looseness", "impact", "shock", "kurtosis", "crest factor"],
        "bearing fault": ["bearing", "BPFO", "BPFI", "ball pass", "inner race", "outer race"],
    }
    
    query_parts = []

    # 1) 어떤 task인지 한 줄로
    query_parts.append(
        f"vibration-based fault diagnosis for rotating machinery "
        f"among {target_labels}"
    )

    # 2) 가장 큰 변화가 있는 feature들
    if extreme_features or large_features:
        top_feats = (extreme_features + large_features)[:8]
        feat_names = [f[0] for f in top_feats]
        query_parts.append(
            "abnormal vibration features: " + ", ".join(feat_names)
        )

    # 3) 의심되는 고장 유형
    if suspected_faults:
        query_parts.append(
            "suspected fault types: " + ", ".join(suspected_faults)
        )

    # 4) 우리가 원하는 것
    query_parts.append(
        "provide diagnostic criteria, thresholds, and rules "
        "to distinguish these fault types based on these features"
    )

    base_query = " | ".join(query_parts)
    
    # 비정상 특징 요약 생성
    abnormal_summary = []
    if extreme_features:
        extreme_summary = ", ".join([f"{f[0]}={f[1]:.1f}%" for f in extreme_features[:3]])
        abnormal_summary.append(f"EXTREME changes (>50%): {extreme_summary}")
    if large_features:
        large_summary = ", ".join([f"{f[0]}={f[1]:.1f}%" for f in large_features[:3]])
        abnormal_summary.append(f"LARGE changes (20-50%): {large_summary}")
    
    # current_knowledge를 문자열로 변환 (표시용)
    if isinstance(current_knowledge, dict):
        knowledge_str = ", ".join([f"{k}: {v:.4f}" for k, v in list(current_knowledge.items())[:20]])
    else:
        knowledge_str = str(current_knowledge)[:400]
    
    query = (
        base_query + ". "
        f"Current feature change ratios (vs normal): {knowledge_str}. "
        "Focus on rules and thresholds for interpreting skewness, kurtosis, orders, RMS, crest factor, "
        "and bearing characteristic frequencies (BPFO/BPFI)."
    )
    
    # retriever의 k 값을 가져오기
    original_k = retriever.search_kwargs.get('k', 4) if hasattr(retriever, 'search_kwargs') else 4
    
    # k + 3개를 검색하기 위해 vectorstore에 직접 접근
    # retriever.vectorstore 또는 retriever._vectorstore 속성 사용
    vectorstore = None
    if hasattr(retriever, 'vectorstore'):
        vectorstore = retriever.vectorstore
    elif hasattr(retriever, '_vectorstore'):
        vectorstore = retriever._vectorstore
    
    if vectorstore is not None:
        # vectorstore에서 직접 더 많은 문서 검색 (MMR 사용)
        retrieved_docs = vectorstore.max_marginal_relevance_search(
            query=query,
            k=original_k + 3,
            fetch_k=(original_k + 3) * 2,
            lambda_mult=0.5
        )
    else:
        # fallback: get_relevant_documents 사용 (k 파라미터 지원 여부 확인)
        if hasattr(retriever, 'get_relevant_documents'):
            try:
                # k 파라미터를 전달해보고, 안 되면 기본 호출
                retrieved_docs = retriever.get_relevant_documents(query, k=original_k + 3)
            except TypeError:
                # k 파라미터를 지원하지 않으면 기본 호출
                retrieved_docs = retriever.get_relevant_documents(query)
        else:
            # 최종 fallback: invoke 사용
            retrieved_docs = retriever.invoke(query)
    
    # 중복 제거: 소스 파일 + 청크 인덱스 조합으로 중복 제거
    seen_keys = set()
    unique_docs = []
    
    for doc in retrieved_docs:
        meta = getattr(doc, "metadata", {}) if hasattr(doc, "metadata") else {}
        source = meta.get('source', 'unknown')
        chunk_idx = meta.get('chunk_index', -1)
        
        # 소스 파일과 청크 인덱스 조합으로 중복 판단
        key = (source, chunk_idx)
        if key not in seen_keys:
            seen_keys.add(key)
            unique_docs.append(doc)
            # 원래 k개가 모이면 중단
            if len(unique_docs) >= original_k:
                break
    
    # 정확히 k개 반환 (부족하면 그대로 반환)
    return unique_docs[:original_k]

def make_retriever(
    embedding_model: str,
    model_cache: str,
    docs_path: str,
    retriever_k: int
):
    """
    벡터 스토어와 retriever 생성
    
    벡터 스토어가 없으면 자동으로 생성하고, 있으면 기존 것을 재사용합니다.
    
    Args:
        embedding_model: 임베딩 모델명
        model_cache: 모델 캐시 경로
        docs_path: 문서 경로
        retriever_k: 검색할 문서 수
    
    Returns:
        Retriever 객체
    """
    persist_directory = os.path.join(docs_path, "vectorstore")
    
    # 임베딩 모델 로딩 (재사용 시에도 필요)
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        encode_kwargs={"normalize_embeddings": True},
        cache_folder=model_cache
    )
    
    # 기존 벡터 스토어가 있으면 재사용
    if os.path.exists(persist_directory):
        print(f"기존 벡터 스토어 로드 중: {persist_directory}")
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        print(f"벡터 스토어 로드 완료 (기존 DB 사용)")
    else:
        # 벡터 스토어가 없으면 생성
        
        # docs_path 폴더에 있는 TXT 파일들을 불러오기
        txt_files = [os.path.join(docs_path, f) for f in os.listdir(docs_path) if f.lower().endswith('.txt')]
        if not txt_files:
            raise ValueError(f"docs_path에 .txt 파일이 없습니다: {docs_path}")
        
        raw_docs = []
        for path in txt_files:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
                raw_docs.append(Document(
                    page_content=text,
                    metadata={'source': os.path.basename(path)}
                ))
        
        print(f"로드된 문서 수: {len(raw_docs)}")
        
        # SemanticTextSplitter로 문서 구조 기반 분할 (메타데이터 강화 포함)
        text_splitter = SemanticTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(raw_docs)
        print(f"생성된 청크 수: {len(docs)}")
        
        # VectorDB에 문서 저장
        print("벡터 스토어 생성 중...")
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        print(f"벡터 스토어 저장 완료: {persist_directory}")
    
    # MMR 검색을 사용하여 Retriever 생성 (검색 다양성 향상)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": retriever_k,
            "fetch_k": retriever_k * 3,  # 다양성 확보를 위해 더 많은 후보 검색
            "lambda_mult": 0.5  # 유사도와 다양성 균형 (0.0=다양성, 1.0=유사도)
        }
    )
    
    return retriever

def format_docs(docs):
    """검색된 문서를 간소화된 형식으로 포맷팅"""
    lines = []
    for idx, doc in enumerate(docs, 1):
        text = getattr(doc, "page_content", str(doc))
        lines.append(f"[DOC{idx}]:\n{text}")
    return "\n\n".join(lines)


class Planner:
    def __init__(self,
                tokenizer,
                llm,
                retreiver,
                max_tokens,
                device: Optional[torch.device] = None,
                ):
        self.llm = llm.eval()
        self.tokenizer = tokenizer
        self.retreiver = retreiver
        self.max_tokens = max_tokens
        self.device = device or torch.device("cpu")
        
        self.target_labels = "normal(healthy), misalignment, looseness, unbalance, bearing fault"
    
    def plan(self, current_knowledge, retrive_docs):
        docs_text = format_docs(retrive_docs)  # 아래에서 설명할 format_docs 활용
        prompt = (
            "System: You are a senior vibration analyst. Be precise and cite sources.\n"
            f"User: We will diagnose rotating machinery among: {self.target_labels}.\n"
            f"Current vibration state (change rates % from normal baseline): {current_knowledge}. "
            "Evidence snippets from manuals/papers are given below, each prefixed with [DOC#]. \n"
            "Extract concrete, actionable rules (thresholds, patterns, symptom descriptions) with citations like [DOC3].\n"
            "Return STRICT JSON with keys: plan_steps, diagnosis_plan.\n"
            f"- plan_steps: 3–5 very short imperative steps (one line each).\n"
            "- diagnosis_plan: object with keys {self.target_labels}.\n"
            "For each key, include at most 2 items.\n"
            'Each item must be: {"diagnosis idea": "<one line>", "why": "<short reason (<=30 tokens)>", "source": "DOC#"}.'
            "Constraints: Use only information supported by the snippets.\n"
            f"Evidence:\n{docs_text}\n"
            "Do NOT invent any new numeric thresholds or specific example values that are not explicitly present in the evidence.\n"
            'If a threshold value is not given, keep it qualitative (e.g., "high", "very high").\n'
            "Assistant: Output JSON only, no extra text. Do not hallucinate."
        )
        with torch.no_grad():
            inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
            input_len = inputs["input_ids"].shape[1]

            out_ids = self.llm.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=False
            )

            # 모델이 새로 생성한 토큰만 디코딩
            gen_ids = out_ids[0, input_len:]
            gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        # 여기서는 JSON만 나오도록 프롬프트를 짰으니 그대로 반환
        return gen_text.strip()
    
    def summerize(self, plan):
        prompt = (
            "System: You are a senior vibration analyst. Create a compact briefing from the given thinking/plan.\n"
            "User: Summarize given PLAN into STRICT JSON with keys: plan_steps, diagnosis_plan.\n"
            "PLAN:\n" + plan + "\n"
            "Assistant: Output JSON only."
        )
        with torch.no_grad():
            input_ids = self.tokenizer(prompt, return_tensors='pt').to(self.device)
            out_ids = self.llm.generate(**input_ids, max_new_tokens=self.max_tokens, do_sample=False)
            out_text = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
        return out_text.strip()
    
    def __call__(self, current_knowledge):
        retrive_docs = retrieve_documents(
            retriever=self.retreiver, 
            current_knowledge=current_knowledge
        )
        plan_json = self.plan(current_knowledge, retrive_docs)
        return plan_json  # 그대로 JSON string
    
    def test_prompt(self, prompt: str) -> str:
        with torch.no_grad():
            inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
            input_len = inputs["input_ids"].shape[1]
            out_ids = self.llm.generate(**inputs, max_new_tokens=self.max_tokens, do_sample=False)
            gen_ids = out_ids[0, input_len:]
            gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return gen_text.strip()
        


class LLM_Dataset(Dataset):
    """
    RAG 검색 결과와 프롬프트를 생성하는 데이터셋 클래스
    
    __getitem__에서 다음을 반환:
    - cur_status: 변화율 딕셔너리
    - x_feat: 현재 상태 특징 딕셔너리
    - ref_feat: 참조 상태 특징 딕셔너리 (있을 경우)
    - rag_docs: Document 객체 리스트
    - prompt: 전체 프롬프트 (System + User + Assistant)
    
    """
    
    def __init__(self,
                 vibration_dataset: VibrationDataset,
                 retriever,
                 planner,
                 target_labels: str = "normal(healthy), misalignment, looseness, unbalance, bearing fault"):
        """
        Args:
            vibration_dataset: VibrationDataset 인스턴스
            retriever: 벡터 검색기
            target_labels: 진단 대상 레이블 문자열
        """
        self.vibration_dataset = vibration_dataset
        self.retriever = retriever
        self.target_labels = target_labels
        self.planner = planner
        
        self.feature_dataset = FeatureExtractLLMDataset(vibration_dataset=vibration_dataset)
        
    def __len__(self):
        return len(self.vibration_dataset)
    
    def _format_change_ratios(self, current_knowledge: Dict[str, float], top_k: int = 20) -> str:
        if not isinstance(current_knowledge, dict):
            return str(current_knowledge)

        # 변화율 절댓값 기준 상위 k개만 출력
        items = sorted(current_knowledge.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]
        return ", ".join([f"{k}: {v:.1f}%" for k, v in items])
    
    def _create_prompt(self, current_knowledge: Dict[str, float], plan: str) -> str:
        """GRPO 학습에 맞게 압축된 프롬프트 생성"""

        # 변화율 딕셔너리를 문자열로 변환
        # (나중에 JSON 포맷으로 바꾸고 싶다면: knowledge_str = json.dumps(current_knowledge, ensure_ascii=False))
        knowledge_str = self._format_change_ratios(current_knowledge, top_k=20)

        # 1) System 프롬프트: 짧고 역할만 명확하게
        system_prompt = (
            "You are an AI diagnostic engine specialized in vibration-based fault detection for rotating machinery. "
            "Follow the given rules and always output a consistent JSON diagnosis. "
            "Use ONLY the provided vibration tokens, feature change ratios, and reference plan. "
            "Do NOT invent extra sensor data or frequencies."
        )

        # 2) User 프롬프트: 핵심 정보 + 규칙 + 출력 형식만 유지
        user_prompt = f"""
        ### TASK
        Classify the current machine state into exactly ONE of:
        normal(healthy), misalignment, looseness, unbalance, bearing fault.

        ### DATA

        1) Vibration tokens:
        - Normal state token: <x_stft>
        - Current state token: <ref_stft>

        2) Feature change ratio (current vs normal):
        {knowledge_str}
        (Positive = increase from normal, Negative = decrease from normal.)

        3) Reference plan & criteria (JSON):
        {plan}

        ### RULES

        A. Token-based rules (Step 1)
        - You CANNOT see the numeric or spectral content of normal and current state tokens. They are abstract tokens.
        - You MUST NOT claim that the tokens are "similar", "different", or "show deviation", because you cannot observe their values.
        - In this setup, treat the token-based analysis as a weak prior:
        - If the plan or features strongly support a specific fault, you MAY set vib_only_label to the SAME fault
            (so that tokens are consistent with the feature-based conclusion).
        - Otherwise, you MAY set vib_only_label = "normal(healthy)" as a neutral prior.
        - You must still pick exactly one label from:
        normal(healthy), misalignment, looseness, unbalance, bearing fault.

        B. Feature-based rules (Step 2)
        Use the feature change ratios above plus the reference plan to choose ONE label.
        - Large positive increases in kurtosis or crest factor (e.g., strong positive change %) → evidence for impulsive events (looseness or bearing fault).
        - Large negative changes in kurtosis, skewness, or crest factor mean "less impulsive" and should NOT be used as evidence for bearing fault.
        - You may ignore features that are not mentioned in the plan unless they are clearly critical.
        
        Step 2 must also output exactly one label from:
        normal(healthy), misalignment, looseness, unbalance, bearing fault.

        C. Fusion rules (Step 3)
        - If Step 1 label and Step 2 label are the same → final_label = that label.
        - If they are different:
        - If strong feature evidence exists (according to the rules above) → trust the feature-based label.
        - Otherwise → trust the token-based label.

        ### INSTRUCTIONS

        Think in three short steps inside the <reasoning> block:

        <reasoning>
        Step 1 (tokens):
        - Explicitly state that you CANNOT observe the numeric content of normal and current state tokens.
        - Then choose vib_only_label according to the weak-prior rule above
        (either copy the feature-based fault when evidence is strong, or use normal(healthy) as a neutral prior).

        Step 2 (features):
        - Use the feature change ratios and the reference plan.
        - Choose knowledge_only_label (one of: normal(healthy), misalignment, looseness, unbalance, bearing fault).
        - State which features drove your choice.

        Step 3 (fusion):
        - Compare vib_only_label and knowledge_only_label.
        - Apply the fusion rules above.
        - Choose final_label and list 2–3 key indicators from the features.
        </reasoning>

        Then output ONLY one JSON object inside an <answer> block:

        <answer>{{
        "vib_only_label": "<one of: normal(healthy), misalignment, looseness, unbalance, bearing fault>",
        "vib_reason": "<1–2 sentences explaining the token-based conclusion WITHOUT claiming to see numeric differences or similarity between tokens",
        "knowledge_only_label": "<one of: normal(healthy), misalignment, looseness, unbalance, bearing fault>",
        "knowledge_reason": "<1–2 sentences explaining the feature-based conclusion>",
        "criteria": [
        "<first key feature indicator>",
        "<second key feature indicator>",
        "<optional third key feature indicator>"
        ],
        "final_label": "<one of: normal(healthy), misalignment, looseness, unbalance, bearing fault>",
        "fusion_reason": "<1–2 sentences explaining how you fused Step 1 and Step 2>"
        }}</answer>
        """

        return f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"
    
    def __getitem__(self, index):

        # 원본 데이터 특징 추출
        data_dict = self.vibration_dataset[index]
        feature_sample = self.feature_dataset[index]
        
        # RAG 검색 및 프롬프트 생성
        cur_status = feature_sample.get('cur_status', {})
        
        plan = self.planner(cur_status)
        
        prompt = self._create_prompt(cur_status, plan)
        
        # prompt test 용
        # llm_response = self.planner.test_prompt(prompt)
        # print(llm_response)

        # 진단 Ground Truth (GRPO accuracy reward 계산용)
        gt_label = feature_sample.get('gt', None)
        
        # 결과 딕셔너리 구성
        return {
            'cur_status': cur_status,
            'x_feat': feature_sample.get('x_feat', {}),
            'ref_feat': feature_sample.get('ref_feat', None),
            'prompt': prompt,
            'gt': gt_label,
            'x_stft': feature_sample.get('x_stft'),
            'ref_stft': feature_sample.get('ref_stft')
        }


def get_llm_dataset(
    args,
    train_dataset,
    val_dataset
):
    """
    Train/Val LLM_Dataset 생성 편의 함수
    
    Args:
        data_root: 데이터셋 루트 경로 (필수)
        train_domain: 학습용 데이터셋 도메인 리스트 (필수, 예: ['vat', 'vbl', 'mfd'])
        valid_domain: 검증용 데이터셋 도메인 리스트 (필수, 예: ['dxai'])
        docs_path: 문서 디렉토리 경로 (기본값: "docs_path")
        window_sec: 윈도우 길이 (초) (기본값: 5.0)
        stride_sec: 스트라이드 길이 (초) (기본값: 3.0)
        include_ref: 참조 데이터 포함 여부 (기본값: True)
        transform: 변환 함수 (STFT 등) (기본값: None)
        embedding_model: 임베딩 모델명 (기본값: "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        model_cache: 모델 캐시 경로 (기본값: None, None이면 ~/.cache/huggingface 사용)
        retriever_k: 검색할 문서 수 (기본값: 4)
        target_labels: 진단 대상 레이블 문자열 (기본값: "normal(healthy), misalignment, looseness, unbalance, bearing fault")
        drop_last: 마지막 배치 버리기 여부 (기본값: True)
        dtype: 데이터 타입 (기본값: None)
        channel_order: 채널 순서 (기본값: ("x", "y"))
        test_mode: 테스트 모드 여부 (기본값: False)
        
    Returns:
        (train_llm_dataset, val_llm_dataset) 튜플
    """
    # Retriever 생성 (train/val 공유)
    retriever = make_retriever(
        embedding_model=args.embedding_model,
        model_cache=args.cache_dir,
        docs_path=args.docs_path,
        retriever_k=args.retriever_k
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.llm_name,
        cache_dir=args.cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    llm = AutoModelForCausalLM.from_pretrained(
                args.llm_name,
                cache_dir=args.cache_dir,
    ).to(device)

    planner = Planner(
        tokenizer=tokenizer,
        llm=llm,
        retreiver=retriever,
        max_tokens=4096,
        device=device,
    )
    
    # create_retrieve_dataset를 사용하여 train/val LLM_Dataset 생성
    target_labels = "normal(healthy), misalignment, looseness, unbalance, bearing fault"
    train_llm_dataset = LLM_Dataset(vibration_dataset=train_dataset, 
                                    retriever=retriever, 
                                    planner=planner,
                                    target_labels=target_labels)
    val_llm_dataset = LLM_Dataset(vibration_dataset=val_dataset, 
                                    retriever=retriever, 
                                    planner=planner,
                                    target_labels=target_labels)
    
    return train_llm_dataset, val_llm_dataset
