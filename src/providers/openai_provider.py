from __future__ import annotations

from dataclasses import dataclass, field

from openai import OpenAI

from prompts.translation import get_translation_prompt


@dataclass
class OpenAITranslationProvider:
    """
    OpenAI translation provider implementation.

    Uses OpenAI's Chat Completions API to translate text.

    Example:
        >>> provider = OpenAITranslationProvider(api_key="sk-...")
    """

    api_key: str
    client: OpenAI = field(init=False)

    key: str = "openai"
    name: str = "openai"
    default_model: str = "gpt-5-nano"
    default_rpm: int = 60

    def __post_init__(self) -> None:
        """Initialize the OpenAI client."""
        self.client = OpenAI(api_key=self.api_key)

    def translate_text(self, text: str, model: str, target_language: str) -> str:
        """
        Translate text using OpenAI Chat Completions.

        Args:
            text: The text to translate.
            model: The model to use (e.g., "gpt-4o").
            target_language: The target language.

        Returns:
            The translated text.
        """
        prompt = get_translation_prompt(None, target_language=target_language, text=text)
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content or ""
        return content.strip()
