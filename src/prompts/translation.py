from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast

if TYPE_CHECKING:
    from core.types import TranslationPromptKey

DEFAULT_TRANSLATION_PROMPT_KEY: Final["TranslationPromptKey"] = "literary_v1"

TRANSLATION_PROMPTS = cast(
    "dict[TranslationPromptKey, str]",
    {
        "literary_v1": (
            "Translate the following text into {target_language} in a literary, artistic manner.\n"
            "Do not add anything, do not modify structure, only translate the meaning:\n\n"
            "{text}"
        ),
    },
)


def get_translation_prompt(key: str | None, *, target_language: str, text: str) -> str:
    """
    Return the translation prompt for the given key.

    Raises:
        ValueError: If the key is unknown.
    """
    prompt_key = key or DEFAULT_TRANSLATION_PROMPT_KEY
    if prompt_key not in TRANSLATION_PROMPTS:
        options = ", ".join(sorted(TRANSLATION_PROMPTS.keys()))
        raise ValueError(
            f"Unknown translation prompt key '{prompt_key}'. Available: {options}."
        )
    template = TRANSLATION_PROMPTS[cast("TranslationPromptKey", prompt_key)]
    return template.format(target_language=target_language, text=text)
