from __future__ import annotations

from typing import Final

DEFAULT_ACTING_INSTRUCTION_PROMPT_KEY: Final[str] = "en_director_v1"

ACTING_INSTRUCTION_PROMPTS: dict[str, str] = {
    "en_director_v1": (
        "You are a voice director and diction specialist.\n\n"
        "Given a text in English, analyze it and produce a brief, precise professional "
        "instruction for speech synthesis that describes HOW this text should be delivered.\n\n"
        "Specify:\n"
        "- type (narration, dialogue, monologue, description, tense scene, calm scene)\n"
        "- emotional tone (calm, tense, dramatic, ironic, etc.)\n"
        "- recommended speaking pace (slow, medium, fast)\n"
        "- pause character (natural, expressive, short, long)\n"
        "- degree of expressiveness (minimal, moderate, high)\n\n"
        "Requirements:\n"
        "- instruction must be in English\n"
        "- keep it brief: 4-8 lines\n"
        "- do not paraphrase the text\n"
        "- do not add new plot\n"
        "- do not mention models, APIs, or technical details\n"
        "- use phrasing suitable for a professional voice actor\n\n"
        "Output format:\n"
        "{\n"
        '  "type": "<type>",\n'
        '  "emotional_tone": "<emotional_tone>",\n'
        '  "speaking_pace": "<recommended_pace>",\n'
        '  "pause_character": "<pause_character>",\n'
        '  "expressiveness": "<expressiveness_level>",\n'
        '  "instruction": "<brief instruction (4-8 lines)>"\n'
        "}\n\n"
        "If a parameter is ambiguous, choose the most likely option based on the text "
        "and fill every field."
    ),
    "ru_director_v1": (
        "Ты — режиссёр озвучивания и специалист по дикции.\n\n"
        "Получив текст на русском языке, проанализируй его и составь краткую, точную "
        "профессиональную инструкцию для синтеза речи, описывающую КАК должен быть "
        "озвучен ЭТОТ текст.\n\n"
        "Укажи:\n"
        "— тип (повествование, диалог, монолог, описание, напряжённая сцена, "
        "спокойная сцена)\n"
        "— эмоциональный тон (спокойный, напряжённый, драматичный, ироничный и т.д.)\n"
        "— рекомендуемый темп речи (медленный, средний, быстрый)\n"
        "— характер пауз (естественные, выразительные, короткие, длинные)\n"
        "— степень выразительности (минимальная, умеренная, высокая)\n\n"
        "Требования:\n"
        "— инструкция на русском языке\n"
        "— кратко: 4–8 строк\n"
        "— не пересказывай текст\n"
        "— не добавляй сюжет\n"
        "— не упоминай модель, API или технические детали\n"
        "— используй формулировки для профессионального диктора\n\n"
        "Формат вывода:\n"
        "{\n"
        '  "тип": "<тип>",\n'
        '  "эмоциональный_тон": "<эмоциональный_тон>",\n'
        '  "темп_речи": "<рекомендуемый_темп>",\n'
        '  "характер_пауз": "<характер_пауз>",\n'
        '  "выразительность": "<степень_выразительности>",\n'
        '  "инструкция": "<краткая инструкция (4-8 строк)>"\n'
        "}\n\n"
        "Пояснения по допустимым значениям параметров приведены ниже. Если нет "
        "однозначного выбора, выбери наиболее вероятный по смыслу текста вариант "
        "и укажи его. Не оставляй поля пустыми."
    ),
}


def get_acting_instruction_prompt(key: str | None) -> str:
    """
    Return the acting-instruction prompt for the given key.

    Raises:
        ValueError: If the key is unknown.
    """
    prompt_key = key or DEFAULT_ACTING_INSTRUCTION_PROMPT_KEY
    try:
        return ACTING_INSTRUCTION_PROMPTS[prompt_key]
    except KeyError as exc:
        options = ", ".join(sorted(ACTING_INSTRUCTION_PROMPTS.keys()))
        raise ValueError(
            f"Unknown acting instruction prompt key '{prompt_key}'. Available: {options}."
        ) from exc
