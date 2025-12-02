# rewards_utils.py
import json
import re
from typing import Dict, Any, List, Tuple, Optional

ALLOWED_LABELS = [
    "normal(healthy)",
    "misalignment",
    "looseness",
    "unbalance",
    "bearing fault",
]

# -------------------- 공통 유틸 -------------------- #

def normalize_label(label: str) -> str:
    """라벨 string을 느슨하게 정규화해서 비교용으로 사용."""
    if not isinstance(label, str):
        return ""
    # 소문자 + 알파벳/숫자만 남기기
    return re.sub(r"[^a-z0-9]", "", label.lower())


def extract_blocks(output_text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    completion 전체에서
    - <reasoning>...</reasoning>
    - <answer> { ... } </answer>
    블록을 추출.
    """
    if output_text is None:
        return None, None

    reasoning_match = re.search(
        r"<reasoning>\s*(.*?)\s*</reasoning>",
        output_text,
        re.DOTALL | re.IGNORECASE,
    )
    answer_match = re.search(
        r"<answer>\s*(\{.*?\})\s*</answer>",
        output_text,
        re.DOTALL | re.IGNORECASE,
    )

    reasoning = reasoning_match.group(1).strip() if reasoning_match else None
    answer_json_str = answer_match.group(1).strip() if answer_match else None
    return reasoning, answer_json_str


def parse_answer_json(answer_json_str: Optional[str]) -> Optional[Dict[str, Any]]:
    """<answer> 내부 JSON을 파싱."""
    if not answer_json_str:
        return None
    try:
        return json.loads(answer_json_str)
    except json.JSONDecodeError:
        # trailing comma 같은 사소한 실수 보정 시도
        try:
            cleaned = re.sub(r",\s*}", "}", answer_json_str)
            cleaned = re.sub(r",\s*]", "]", cleaned)
            return json.loads(cleaned)
        except Exception:
            return None


def get_gt_and_features(gt_obj: object) -> Tuple[Optional[str], Dict[str, float]]:
    """
    gts 리스트에 들어오는 gt 형태를 유연하게 처리:
    - gt_obj 가 str 이면 => (label, {})
    - gt_obj 가 dict 이고 {"gt": ..., "cur_status": {...}} 형태면 둘 다 반환
    """
    if isinstance(gt_obj, str):
        return gt_obj, {}
    if isinstance(gt_obj, dict):
        label = gt_obj.get("gt", None)
        cur_status = gt_obj.get("cur_status", {}) or {}
        if not isinstance(cur_status, dict):
            cur_status = {}
        return label, cur_status
    return None, {}


def format_reward(
    prompts: List[str],
    completions: List[str],
    gts: List[object],
) -> List[float]:
    """
    1. <answer> 블록 존재
    2. JSON 파싱 성공
    3. 필수 키(vib_only_label, knowledge_only_label, final_label 등) 포함
    모두 만족하면 1.0, 아니면 0.0
    """
    scores: List[float] = []

    required_keys = [
        "vib_only_label",
        "vib_reason",
        "knowledge_only_label",
        "knowledge_reason",
        "criteria",
        "final_label",
        "fusion_reason",
    ]

    for completion in completions:
        reasoning, answer_json_str = extract_blocks(completion)
        answer = parse_answer_json(answer_json_str)

        if answer is None:
            scores.append(0.0)
            continue

        ok = True
        for k in required_keys:
            if k not in answer:
                ok = False
                break

        scores.append(1.0 if ok else 0.0)

    return scores


def accuracy_reward(
    prompts: List[str],
    completions: List[str],
    gts: List[object],
) -> List[float]:
    """
    final_label vs gt
    - 같으면 1.0
    - 다르면 0.0
    """
    scores: List[float] = []

    for completion, gt_obj in zip(completions, gts):
        gt_label, _ = get_gt_and_features(gt_obj)
        if gt_label is None:
            scores.append(0.0)
            continue

        _, answer_json_str = extract_blocks(completion)
        answer = parse_answer_json(answer_json_str)
        if answer is None:
            scores.append(0.0)
            continue

        final_label = answer.get("final_label", "")
        if not isinstance(final_label, str):
            scores.append(0.0)
            continue

        if normalize_label(final_label) == normalize_label(gt_label):
            scores.append(1.0)
        else:
            scores.append(0.0)

    return scores


def fusion_reward(
    prompts: List[str],
    completions: List[str],
    gts: List[object],
) -> List[float]:
    """
    final_label 과 knowledge_only_label 이 같으면 1.0, 아니면 0.0
    """
    scores: List[float] = []

    for completion in completions:
        _, answer_json_str = extract_blocks(completion)
        answer = parse_answer_json(answer_json_str)
        if answer is None:
            scores.append(0.0)
            continue

        final_label = answer.get("final_label", "")
        k_label = answer.get("knowledge_only_label", "")
        if not isinstance(final_label, str) or not isinstance(k_label, str):
            scores.append(0.0)
            continue

        scores.append(
            1.0 if normalize_label(final_label) == normalize_label(k_label) else 0.0
        )

    return scores

def feature_usage_reward(
    prompts: List[str],
    completions: List[str],
    gts: List[object],
    top_k: int = 5,
) -> List[float]:
    """
    cur_status에서 |변화율|이 큰 상위 top_k feature 들이
    - criteria
    - knowledge_reason
    에 얼마나 등장하는지 비율로 보상 (0~1).
    gts[i] 가 dict 이고, "cur_status" 키를 포함해야 제대로 동작.
    """
    scores: List[float] = []

    for completion, gt_obj in zip(completions, gts):
        _, cur_status = get_gt_and_features(gt_obj)
        if not cur_status:
            scores.append(0.0)
            continue

        _, answer_json_str = extract_blocks(completion)
        answer = parse_answer_json(answer_json_str)
        if answer is None:
            scores.append(0.0)
            continue

        # 상위 |change| top_k feature
        sorted_items = sorted(
            cur_status.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        top_features = [name for name, _ in sorted_items[:top_k]]

        knowledge_reason = str(answer.get("knowledge_reason", "")).lower()
        criteria = answer.get("criteria", "")

        if isinstance(criteria, list):
            criteria_text = " ".join(str(c) for c in criteria).lower()
        else:
            criteria_text = str(criteria).lower()

        text = knowledge_reason + " " + criteria_text
        if not text.strip():
            scores.append(0.0)
            continue

        cnt = 0
        for feat in top_features:
            if feat.lower() in text:
                cnt += 1

        if top_k <= 0:
            scores.append(0.0)
        else:
            scores.append(cnt / float(top_k))

    return scores


FORBIDDEN_PATTERNS = [
    r"<x_stft>.*similar",
    r"<ref_stft>.*similar",
    r"<x_stft>.*different",
    r"<ref_stft>.*different",
    r"<x_stft>.*deviation",
    r"<ref_stft>.*deviation",
    r"similar.*<x_stft>",
    r"different.*<ref_stft>",
    r"mismatch.*<x_stft>",
    r"mismatch.*<ref_stft>",
]

def no_hallucination_reward(
    prompts: List[str],
    completions: List[str],
    gts: List[object],
) -> List[float]:
    """
    토큰에 대해 'similar/different/deviation' 등 금지 표현이 나오면 0.0
    없으면 1.0
    """
    scores: List[float] = []

    for completion in completions:
        reasoning, answer_json_str = extract_blocks(completion)
        text_parts = []
        if reasoning:
            text_parts.append(reasoning)
        # vib_reason 도 같이 체크
        answer = parse_answer_json(answer_json_str)
        if answer is not None and "vib_reason" in answer:
            text_parts.append(str(answer["vib_reason"]))

        text = " ".join(text_parts).lower()

        violated = False
        for pat in FORBIDDEN_PATTERNS:
            if re.search(pat, text):
                violated = True
                break

        scores.append(0.0 if violated else 1.0)

    return scores

def structure_reward(
    prompts: List[str],
    completions: List[str],
    gts: List[object],
) -> List[float]:
    """
    <reasoning> 안에 Step 1/2/3 이 얼마나 잘 나오는지:
    - 세 개 다 있으면 1.0
    - 일부만 있으면 0.5
    - 없으면 0.0
    """
    scores: List[float] = []

    for completion in completions:
        reasoning, _ = extract_blocks(completion)
        if not reasoning:
            scores.append(0.0)
            continue

        lower = reasoning.lower()
        has_1 = "step 1" in lower
        has_2 = "step 2" in lower
        has_3 = "step 3" in lower

        count = int(has_1) + int(has_2) + int(has_3)
        if count == 3:
            scores.append(1.0)
        elif count >= 1:
            scores.append(0.5)
        else:
            scores.append(0.0)

    return scores