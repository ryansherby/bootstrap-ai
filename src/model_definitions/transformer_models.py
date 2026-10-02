from configs import CONFIG
import re

from google import genai
from google.genai.errors import APIError

from model_definitions.gemini_model import GeminiModel
from model_definitions.retry_utils import sleep_for_retry


def _render_prompt(
    prompt_path: str,
    replacements: dict[str, object],
) -> str:
    with open(
        prompt_path,
        "r",
        encoding="utf-8",
    ) as f:
        prompt = f.read()

    for key, value in replacements.items():
        prompt = prompt.replace(
            f"{{{{{key}}}}}",
            "" if value is None else str(value),
        )

    unresolved = sorted(
        set(
            re.findall(
                r"\{\{[A-Z0-9_]+\}\}",
                prompt,
            )
        )
    )
    if unresolved:
        raise ValueError(
            "Unresolved prompt placeholders in "
            f"{prompt_path}: {unresolved}"
        )

    return prompt


def _strip_markdown_fences(text: str) -> str:
    cleaned = text.strip()

    if "```" not in cleaned:
        return cleaned

    match = re.search(
        r"```(?:python)?\s*(.*?)```",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()

    return cleaned.replace("```", "").strip()


class GeminiFlashModel(GeminiModel):

    name = "gemini"

    MAX_RETRIES = CONFIG.transformer_model.max_retries

    def __init__(self, gpu_number=0, device="cuda"):
        super().__init__(gpu_number, device)

        self.model = "gemini-2.5-flash"
        self.temperature = CONFIG.transformer_model.temperature

    @classmethod
    def list_processes(cls):
        return [
            "code",
            "merge",
            "plan",
            "comparison_rules",
            "test",
        ]

    def _generate(self, prompt: str) -> str:
        for attempt in range(self.MAX_RETRIES + 1):

            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        temperature=self.temperature,
                    ),
                )

                return response.text

            except APIError as e:

                if e.code != 429:
                    raise

                if attempt >= self.MAX_RETRIES:
                    raise

                response = getattr(e, "response", None)

                headers = (
                    response.headers
                    if response is not None
                    else None
                )

                sleep_for_retry(
                    headers=headers,
                    attempt=attempt,
                )

        raise RuntimeError(
            "Unreachable: Gemini retry loop exited unexpectedly."
        )

    def generate_code(
        self,
        table_metadata,
        functions,
        query,
        plan,
    ):
        prompt = _render_prompt(
            CONFIG.transformer_model.code_prompt_path,
            {
                "TABLE_METADATA": table_metadata,
                "AVAILABLE_FUNCTIONS": functions,
                "QUERY": query,
                "PLAN": plan,
            },
        )

        return _strip_markdown_fences(self._generate(prompt))

    def merge_code(self, code_snippets):
        if len(code_snippets) < 2:
            raise ValueError(
                "merge requires two code snippets"
            )

        prompt = _render_prompt(
            CONFIG.transformer_model.merge_prompt_path,
            {
                "SOLUTION_A": code_snippets[0],
                "SOLUTION_B": code_snippets[1],
            },
        )

        return _strip_markdown_fences(self._generate(prompt))

    def generate_plan(
        self,
        table_metadata,
        query,
    ):
        prompt = _render_prompt(
            CONFIG.transformer_model.plan_prompt_path,
            {
                "TABLE_METADATA": table_metadata,
                "QUERY": query,
            },
        )

        return _strip_markdown_fences(self._generate(prompt))

    def generate_comparison_rules(
        self,
        content,
    ):
        prompt = _render_prompt(
            CONFIG.transformer_model.comparison_rules_prompt_path,
            {
                "CONTENT": content,
            },
        )

        return _strip_markdown_fences(self._generate(prompt))

    def generate_test(
        self,
        code_snippets,
    ):
        prompt = _render_prompt(
            CONFIG.transformer_model.test_prompt_path,
            {
                "CODE_SNIPPETS": "\n".join(code_snippets),
            },
        )

        return _strip_markdown_fences(self._generate(prompt))

    def forward(
        self,
        process_name,
        table_metadata=None,
        functions=None,
        query=None,
        code_snippets=None,
        plan=None,
        comparison_basis=None,
    ):
        if code_snippets is None:
            code_snippets = []

        if process_name == "code":
            return self.generate_code(
                table_metadata,
                functions,
                query,
                plan,
            )

        elif process_name == "merge":
            return self.merge_code(code_snippets)

        elif process_name == "plan":
            return self.generate_plan(
                table_metadata,
                query,
            )

        elif process_name == "comparison_rules":
            return self.generate_comparison_rules(
                comparison_basis,
            )

        elif process_name == "test":
            return self.generate_test(
                code_snippets,
            )

        else:
            raise ValueError(
                f"Unsupported process name: {process_name}"
            )
        
        
        
from model_definitions.openai_model import OpenAIModel

from openai import RateLimitError


class GPT54MiniModel(OpenAIModel):

    name = "gpt"

    MAX_RETRIES = CONFIG.transformer_model.max_retries

    def __init__(
        self,
        gpu_number=0,
        device="cuda",
    ):
        super().__init__(gpu_number, device)

        self.model = "gpt-5.4-mini"
        self.temperature = CONFIG.transformer_model.temperature

    @classmethod
    def list_processes(cls):
        return [
            "code",
            "merge",
            "plan",
            "comparison_rules",
            "test",
        ]

    def _generate(
        self,
        prompt: str,
    ) -> str:

        for attempt in range(self.MAX_RETRIES + 1):

            try:
                response = self.client.responses.create(
                    model=self.model,
                    input=prompt,
                    temperature=self.temperature,
                    reasoning={
                        "effort": "none",
                    },
                )

                return response.output_text

            except RateLimitError as e:

                if attempt >= self.MAX_RETRIES:
                    raise

                response = getattr(e, "response", None)

                headers = (
                    response.headers
                    if response is not None
                    else None
                )

                sleep_for_retry(
                    headers=headers,
                    attempt=attempt,
                )

        raise RuntimeError(
            "Unreachable: OpenAI retry loop exited unexpectedly."
        )

    def generate_code(
        self,
        table_metadata,
        functions,
        query,
        plan,
    ):
        prompt = _render_prompt(
            CONFIG.transformer_model.code_prompt_path,
            {
                "TABLE_METADATA": table_metadata,
                "AVAILABLE_FUNCTIONS": functions,
                "QUERY": query,
                "PLAN": plan,
            },
        )

        return _strip_markdown_fences(self._generate(prompt))

    def merge_code(
        self,
        code_snippets,
    ):
        if len(code_snippets) < 2:
            raise ValueError(
                "merge requires two code snippets"
            )

        prompt = _render_prompt(
            CONFIG.transformer_model.merge_prompt_path,
            {
                "SOLUTION_A": code_snippets[0],
                "SOLUTION_B": code_snippets[1],
            },
        )

        return _strip_markdown_fences(self._generate(prompt))

    def generate_plan(
        self,
        table_metadata,
        query,
    ):
        prompt = _render_prompt(
            CONFIG.transformer_model.plan_prompt_path,
            {
                "TABLE_METADATA": table_metadata,
                "QUERY": query,
            },
        )

        return _strip_markdown_fences(self._generate(prompt))

    def generate_comparison_rules(
        self,
        content,
    ):
        prompt = _render_prompt(
            CONFIG.transformer_model.comparison_rules_prompt_path,
            {
                "CONTENT": content,
            },
        )

        return _strip_markdown_fences(self._generate(prompt))

    def generate_test(
        self,
        code_snippets,
    ):
        prompt = _render_prompt(
            CONFIG.transformer_model.test_prompt_path,
            {
                "CODE_SNIPPETS": "\n".join(code_snippets),
            },
        )

        return _strip_markdown_fences(self._generate(prompt))

    def forward(
        self,
        process_name,
        table_metadata=None,
        functions=None,
        query=None,
        code_snippets=None,
        plan=None,
        comparison_basis=None,
    ):
        if code_snippets is None:
            code_snippets = []

        if process_name == "code":
            return self.generate_code(
                table_metadata,
                functions,
                query,
                plan,
            )

        elif process_name == "merge":
            return self.merge_code(code_snippets)

        elif process_name == "plan":
            return self.generate_plan(
                table_metadata,
                query,
            )

        elif process_name == "comparison_rules":
            return self.generate_comparison_rules(
                comparison_basis,
            )

        elif process_name == "test":
            return self.generate_test(
                code_snippets,
            )

        else:
            raise ValueError(
                f"Unsupported process name: {process_name}"
            )