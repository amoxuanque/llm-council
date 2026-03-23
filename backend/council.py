"""3-stage LLM Council orchestration."""

import re
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from openrouter import query_models_parallel, query_model
from config import COUNCIL_MODELS, CHAIRMAN_MODEL, TITLE_MODEL


def _format_response_for_review(result: Dict[str, Any]) -> str:
    """
    Format a model's response for peer review, including reasoning content
    from thinking models (e.g. DeepSeek-R1) so reviewers see the full output.
    """
    parts = []
    reasoning = result.get('reasoning_content', '')
    content = result.get('response', '')

    if reasoning:
        parts.append(f"<thinking>\n{reasoning}\n</thinking>\n")
    if content:
        parts.append(content)

    return '\n'.join(parts) if parts else content


async def stage1_collect_responses(user_query: str) -> List[Dict[str, Any]]:
    """
    Stage 1: Collect individual responses from all council models.

    Returns:
        List of dicts with 'model', 'response', and optionally 'reasoning_content' keys
    """
    messages = [{"role": "user", "content": user_query}]
    responses = await query_models_parallel(COUNCIL_MODELS, messages)

    stage1_results = []
    for model, response in responses.items():
        if response is not None:
            entry = {
                "model": model,
                "response": response.get('content', ''),
            }
            # Preserve reasoning content from thinking models (R1, etc.)
            if response.get('reasoning_content'):
                entry['reasoning_content'] = response['reasoning_content']
            stage1_results.append(entry)

    return stage1_results


async def stage2_collect_rankings(
    user_query: str,
    stage1_results: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Stage 2: Each model ranks the anonymized responses (excluding its own).

    Returns:
        Tuple of (rankings list, label_to_model mapping)
    """
    labels = [chr(65 + i) for i in range(len(stage1_results))]

    # Create mapping from label to model name
    label_to_model = {
        f"Response {label}": result['model']
        for label, result in zip(labels, stage1_results)
    }

    # Build model-to-label reverse mapping for self-exclusion
    model_to_label = {result['model']: f"Response {label}"
                      for label, result in zip(labels, stage1_results)}

    # For each evaluator, build a prompt excluding its own response
    eval_tasks = []
    eval_models = []

    for evaluator_model in COUNCIL_MODELS:
        # Determine which responses to show (exclude self)
        visible_responses = []
        for label, result in zip(labels, stage1_results):
            if result['model'] == evaluator_model:
                continue  # Skip own response
            full_text = _format_response_for_review(result)
            visible_responses.append(f"Response {label}:\n{full_text}")

        if not visible_responses:
            continue  # Edge case: model's only response

        responses_text = "\n\n".join(visible_responses)

        # List which labels are being evaluated
        visible_labels = [f"Response {label}" for label, result in zip(labels, stage1_results)
                          if result['model'] != evaluator_model]
        labels_hint = "、".join(visible_labels)

        ranking_prompt = f"""你正在评估以下问题的不同回答：

问题：{user_query}

以下是来自不同模型的回答（已匿名处理）：

{responses_text}

你的任务：
1. 首先逐一评估每个回答。对每个回答，说明其优点和不足。
2. 然后在回复的最末尾，给出最终排名。

请用中文撰写你的评估内容。

重要：最终排名必须严格按照以下格式：
- 以 "FINAL RANKING:" 这一行开头（全大写，带冒号）
- 然后按从好到差的顺序用编号列表列出回答
- 每行格式为：数字、句点、空格，然后只写回答标签（例如 "1. Response A"）
- 排名部分不要添加任何其他文字或说明
- 你需要排名的回答有：{labels_hint}

你的完整回复格式示例：

回答A在X方面提供了详细信息，但遗漏了Y...
回答B内容准确但在Z方面缺乏深度...

FINAL RANKING:
1. Response A
2. Response B

请提供你的评估和排名："""

        messages = [{"role": "user", "content": ranking_prompt}]
        eval_tasks.append(query_model(evaluator_model, messages))
        eval_models.append(evaluator_model)

    # Execute all evaluation queries in parallel
    import asyncio
    eval_responses = await asyncio.gather(*eval_tasks)

    stage2_results = []
    for model, response in zip(eval_models, eval_responses):
        if response is not None:
            full_text = response.get('content', '')
            parsed = parse_ranking_from_text(full_text)
            stage2_results.append({
                "model": model,
                "ranking": full_text,
                "parsed_ranking": parsed
            })

    return stage2_results, label_to_model


async def stage3_synthesize_final(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    stage2_results: List[Dict[str, Any]],
    aggregate_rankings: List[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Stage 3: Chairman (non-council member) synthesizes final response.
    """
    stage1_text = "\n\n".join([
        f"Model: {result['model']}\nResponse: {result['response']}"
        for result in stage1_results
    ])

    stage2_text = "\n\n".join([
        f"Model: {result['model']}\nRanking: {result['ranking']}"
        for result in stage2_results
    ])

    consensus_text = ""
    if aggregate_rankings:
        ranked_lines = "\n".join([
            f"{i+1}. {agg['model'].split('/')[-1]} (avg rank: {agg['average_rank']:.2f}, votes: {agg['rankings_count']})"
            for i, agg in enumerate(aggregate_rankings)
        ])
        consensus_text = f"""
PEER CONSENSUS (aggregate rankings, lower score = better):
{ranked_lines}

"""

    chairman_prompt = f"""你是LLM议会的主席。多个AI模型已经对用户的问题提供了回答，并且相互对彼此的回答进行了排名。

原始问题：{user_query}
{consensus_text}
阶段一 - 各模型独立回答：
{stage1_text}

阶段二 - 同行排名：
{stage2_text}

作为主席，你的任务是将所有这些信息综合成一个针对用户原始问题的单一、全面、准确的答案。请考虑：
- 各个回答及其见解
- 同行共识排名作为回答质量的信号
- 各模型之间的共识或分歧模式

请用中文提供一个清晰、有理有据的最终答案，代表议会的集体智慧："""

    messages = [{"role": "user", "content": chairman_prompt}]
    response = await query_model(CHAIRMAN_MODEL, messages)

    if response is None:
        return {
            "model": CHAIRMAN_MODEL,
            "response": "Error: Unable to generate final synthesis."
        }

    return {
        "model": CHAIRMAN_MODEL,
        "response": response.get('content', '')
    }


def parse_ranking_from_text(ranking_text: str) -> List[str]:
    """
    Parse the FINAL RANKING section from the model's response.
    Handles multiple formats including numbered lists and Chinese variations.
    """
    # Try "FINAL RANKING:" section first
    ranking_section = None
    for marker in ["FINAL RANKING:", "Final Ranking:", "FINAL_RANKING:", "最终排名：", "最终排名:"]:
        if marker in ranking_text:
            parts = ranking_text.split(marker)
            if len(parts) >= 2:
                ranking_section = parts[-1]  # Take the last occurrence
                break

    if ranking_section:
        # Try numbered list format: "1. Response A" or "1. Response A"
        numbered_matches = re.findall(r'\d+[\.\、]\s*Response\s+[A-Z]', ranking_section)
        if numbered_matches:
            return [re.search(r'Response\s+[A-Z]', m).group().replace('  ', ' ')
                    for m in numbered_matches]

        # Try Chinese format: "1. 回答A" or "1、回答 A"
        cn_matches = re.findall(r'\d+[\.\、]\s*回答\s*[A-Z]', ranking_section)
        if cn_matches:
            return [f"Response {re.search(r'[A-Z]', m).group()}" for m in cn_matches]

        # Fallback: extract all "Response X" patterns in the section
        matches = re.findall(r'Response\s+[A-Z]', ranking_section)
        if matches:
            return [m.replace('  ', ' ') for m in matches]

    # Last resort: find any "Response X" patterns in entire text
    matches = re.findall(r'Response\s+[A-Z]', ranking_text)
    return [m.replace('  ', ' ') for m in matches]


def calculate_aggregate_rankings(
    stage2_results: List[Dict[str, Any]],
    label_to_model: Dict[str, str]
) -> List[Dict[str, Any]]:
    """
    Calculate aggregate rankings across all models.
    """
    model_positions = defaultdict(list)

    for ranking in stage2_results:
        parsed_ranking = ranking.get('parsed_ranking') or parse_ranking_from_text(ranking['ranking'])

        for position, label in enumerate(parsed_ranking, start=1):
            # Normalize label format
            normalized = label.strip()
            if normalized in label_to_model:
                model_name = label_to_model[normalized]
                model_positions[model_name].append(position)

    aggregate = []
    for model, positions in model_positions.items():
        if positions:
            avg_rank = sum(positions) / len(positions)
            aggregate.append({
                "model": model,
                "average_rank": round(avg_rank, 2),
                "rankings_count": len(positions)
            })

    aggregate.sort(key=lambda x: x['average_rank'])
    return aggregate


async def generate_conversation_title(user_query: str) -> str:
    """
    Generate a short title for a conversation based on the first user message.
    """
    title_prompt = f"""Generate a very short title (3-5 words maximum) that summarizes the following question.
The title should be concise and descriptive. Do not use quotes or punctuation in the title.

Question: {user_query}

Title:"""

    messages = [{"role": "user", "content": title_prompt}]
    response = await query_model(TITLE_MODEL, messages, timeout=30.0)

    if response is None:
        return "New Conversation"

    title = response.get('content', 'New Conversation').strip()
    title = title.strip('"\'')

    if len(title) > 50:
        title = title[:47] + "..."

    return title


async def run_full_council(user_query: str) -> Tuple[List, List, Dict, Dict]:
    """
    Run the complete 3-stage council process.
    """
    # Stage 1
    stage1_results = await stage1_collect_responses(user_query)

    if not stage1_results:
        return [], [], {
            "model": "error",
            "response": "All models failed to respond. Please try again."
        }, {}

    # Stage 2
    stage2_results, label_to_model = await stage2_collect_rankings(user_query, stage1_results)

    # Aggregate
    aggregate_rankings = calculate_aggregate_rankings(stage2_results, label_to_model)

    # Stage 3
    stage3_result = await stage3_synthesize_final(
        user_query, stage1_results, stage2_results, aggregate_rankings
    )

    metadata = {
        "label_to_model": label_to_model,
        "aggregate_rankings": aggregate_rankings
    }

    return stage1_results, stage2_results, stage3_result, metadata
