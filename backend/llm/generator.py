"""LLM integration with prompt injection protection and streaming support."""

from typing import Dict, Generator, List, Optional

from config import Config
from logger import get_logger
from prompting.context import ContextBuilder
from prompting.prompts import SYSTEM_PROMPT, build_rag_prompt

logger = get_logger(__name__)

# Deprecated on many API keys; kept for explicit override detection only
_DEPRECATED_GEMINI_MODELS = frozenset({"gemini-1.5-pro", "gemini-1.5-pro-latest"})


def _gemini_models_to_try() -> list[str]:
    """Build ordered list of Gemini model IDs to attempt."""
    candidates = [Config.GEMINI_MODEL]
    candidates.extend(
        m.strip()
        for m in Config.GEMINI_MODEL_FALLBACKS.split(",")
        if m.strip()
    )
    ordered: list[str] = []
    seen: set[str] = set()
    for name in candidates:
        if name in _DEPRECATED_GEMINI_MODELS:
            logger.warning(
                "GEMINI_MODEL=%s is deprecated; trying fallbacks instead", name
            )
            continue
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    if not ordered:
        ordered = ["gemini-2.0-flash", "gemini-1.5-flash"]
    return ordered


class LLMGenerator:
    """Generates answers using configured LLM provider with RAG context."""

    def __init__(self):
        self.provider = Config.LLM_PROVIDER.lower()
        self.model = None
        self.client = None

        if self.provider == "gemini":
            self._init_gemini()
        elif self.provider == "openai":
            self._init_openai()
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def _init_gemini(self) -> None:
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise ImportError(
                "google-generativeai not installed. Run: pip install google-generativeai"
            ) from exc

        if not Config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment")

        genai.configure(api_key=Config.GEMINI_API_KEY)
        self._genai = genai
        self._gemini_request_options = {
            "timeout": Config.LLM_REQUEST_TIMEOUT_SECONDS,
        }
        self._gemini_model_name: str | None = None
        self.model = self._create_gemini_model()
        logger.info(
            "Initialized Gemini model: %s (timeout=%ss)",
            self._gemini_model_name,
            Config.LLM_REQUEST_TIMEOUT_SECONDS,
        )

    def _create_gemini_model(self):
        """Create Gemini client for the configured (or first fallback) model."""
        model_name = _gemini_models_to_try()[0]
        self._gemini_model_name = model_name
        return self._genai.GenerativeModel(
            model_name,
            system_instruction=SYSTEM_PROMPT,
        )

    def _models_for_request(self) -> list[str]:
        """Models to try for this request; never reuse deprecated cached IDs."""
        models = _gemini_models_to_try()
        cached = self._gemini_model_name
        if cached and cached not in _DEPRECATED_GEMINI_MODELS and cached in models:
            return [cached] + [m for m in models if m != cached]
        return models

    def _gemini_generate_with_fallback(self, *, stream: bool, prompt: str):
        """Call generate_content, retrying with fallback models on 404."""
        models = self._models_for_request()

        last_error: Exception | None = None
        for model_name in models:
            try:
                if model_name != self._gemini_model_name:
                    self.model = self._genai.GenerativeModel(
                        model_name,
                        system_instruction=SYSTEM_PROMPT,
                    )
                    self._gemini_model_name = model_name
                    logger.info("Switched to Gemini model: %s", model_name)

                return self.model.generate_content(
                    prompt,
                    stream=stream,
                    request_options=self._gemini_request_options,
                )
            except Exception as exc:
                last_error = exc
                err = str(exc).lower()
                if any(
                    token in err
                    for token in ("404", "not found", "429", "quota", "resourceexhausted")
                ):
                    logger.warning("Gemini model %s failed: %s", model_name, exc)
                    self._gemini_model_name = None
                    continue
                raise

        raise RuntimeError(
            f"Gemini generation failed after trying {models}: {last_error}"
        ) from last_error

    def _init_openai(self) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("openai not installed. Run: pip install openai") from exc

        if not Config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment")

        self.client = OpenAI(
            api_key=Config.OPENAI_API_KEY,
            timeout=Config.LLM_REQUEST_TIMEOUT_SECONDS,
        )
        self.model = Config.OPENAI_MODEL
        logger.info(
            "Initialized OpenAI model: %s (timeout=%ss)",
            self.model,
            Config.LLM_REQUEST_TIMEOUT_SECONDS,
        )

    def generate_answer(
        self,
        query: str,
        context_chunks: List[str],
        metadata: Optional[List[Dict]] = None,
    ) -> str:
        """Generate a complete answer (non-streaming)."""
        prompt = self._build_prompt(query, context_chunks, metadata)

        if self.provider == "gemini":
            return self._generate_gemini(prompt)
        return self._generate_openai(prompt)

    def generate_answer_stream(
        self,
        query: str,
        context_chunks: List[str],
        metadata: Optional[List[Dict]] = None,
    ) -> Generator[str, None, None]:
        """Stream answer tokens for Streamlit st.write_stream."""
        prompt = self._build_prompt(query, context_chunks, metadata)

        if self.provider == "gemini":
            yield from self._stream_gemini(prompt)
        else:
            yield from self._stream_openai(prompt)

    def _build_prompt(
        self,
        query: str,
        context_chunks: List[str],
        metadata: Optional[List[Dict]] = None,
    ) -> str:
        context_text = ContextBuilder.build(context_chunks, metadata)
        return build_rag_prompt(query, context_text)

    def _generate_gemini(self, prompt: str) -> str:
        try:
            response = self._gemini_generate_with_fallback(stream=False, prompt=prompt)
            return response.text or ""
        except Exception as exc:
            logger.error("Gemini API error: %s", exc)
            raise RuntimeError(f"Gemini generation failed: {exc}") from exc

    def _generate_openai(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1000,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            logger.error("OpenAI API error: %s", exc)
            raise RuntimeError(f"OpenAI generation failed: {exc}") from exc

    def _stream_gemini(self, prompt: str) -> Generator[str, None, None]:
        try:
            response = self._gemini_generate_with_fallback(stream=True, prompt=prompt)
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as exc:
            logger.error("Gemini streaming error: %s", exc)
            msg = str(exc)
            if "404" in msg and "gemini-1.5-pro" in msg:
                msg += (
                    " Restart the app (Ctrl+C, then rerun) so the LLM reloads — "
                    "your session may still be using the retired gemini-1.5-pro model."
                )
            raise RuntimeError(f"Gemini streaming failed: {msg}") from exc

    def _stream_openai(self, prompt: str) -> Generator[str, None, None]:
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1000,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as exc:
            logger.error("OpenAI streaming error: %s", exc)
            raise RuntimeError(f"OpenAI streaming failed: {exc}") from exc
