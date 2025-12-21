from __future__ import annotations

from dataclasses import dataclass, field

from openai import OpenAI


@dataclass
class OpenAITranslationProvider:
    api_key: str
    client: OpenAI = field(init=False)

    key: str = "openai"
    name: str = "openai"
    default_model: str = "gpt-5-nano"
    default_rpm: int = 60

    def __post_init__(self) -> None:
        self.client = OpenAI(api_key=self.api_key)

    def translate_text(self, text: str, model: str, target_language: str) -> str:
        prompt = f"""
Translate the following text into {target_language} in a literary, artistic manner.
Do not add anything, do not modify structure, only translate the meaning:

{text}
""".strip()
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content or ""
        return content.strip()
