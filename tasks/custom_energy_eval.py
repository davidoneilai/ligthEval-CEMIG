from inspect_ai.dataset import Sample
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc

MAX_CONTEXT_CHARS = 4000


def _extract_choices(line: dict):
    labels = line["choices"]["label"]
    texts = line["choices"]["text"]
    return labels, texts


def _sanitize_context(raw_context: str) -> str:
    context = " ".join(raw_context.split())
    if len(context) <= MAX_CONTEXT_CHARS:
        return context
    return context[:MAX_CONTEXT_CHARS].rstrip() + " (...)"


def _build_prompt(question: str, context: str, labels: list[str], texts: list[str]) -> str:
    options = "\n".join(f"({lab}): {txt}" for lab, txt in zip(labels, texts))

    if context:
        return (
            "Voce e um assistente prestativo. Responda de forma direta e objetiva.\n\n"
            f"Contexto:\n{context}\n\n"
            f"Pergunta:\n{question}\n\n"
            f"Alternativas:\n{options}\n\n"
            "Responda apenas com a letra da alternativa correta.\n"
            "Resposta: "
        )

    return (
        "Voce e um assistente prestativo. Responda de forma direta e objetiva.\n\n"
        f"Pergunta:\n{question}\n\n"
        f"Alternativas:\n{options}\n\n"
        "Responda apenas com a letra da alternativa correta.\n"
        "Resposta: "
    )


def energy_eval_prompt(line: dict, task_name: str | None = None) -> Doc:
    labels, texts = _extract_choices(line)
    context = _sanitize_context((line.get("right_context") or "").strip())
    prompt = _build_prompt(
        question=line["question"],
        context=context,
        labels=labels,
        texts=texts,
    )

    gold_index = labels.index(line["answerKey"])

    return Doc(
        task_name=task_name,
        query=prompt,
        choices=labels,  # A, B, C, D, E
        gold_index=gold_index,
    )


def record_to_sample(record: dict) -> Sample:
    labels, texts = _extract_choices(record)
    context = _sanitize_context((record.get("right_context") or "").strip())
    prompt = _build_prompt(
        question=record["question"],
        context=context,
        labels=labels,
        texts=texts,
    )

    return Sample(
        input=prompt,
        target=record["answerKey"],
        choices=labels,
    )


energy_eval = LightevalTaskConfig(
    name="energy_eval",
    prompt_function=energy_eval_prompt,
    hf_repo="cemig-ceia/energy-eval",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

TASKS_TABLE = [energy_eval]