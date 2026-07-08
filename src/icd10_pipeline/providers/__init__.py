from .base import LLMProvider

PROVIDERS = ("anthropic", "openai", "local")


def get_provider(name: str, model: str | None = None) -> LLMProvider:
    """Factory. Imports lazily so missing optional SDKs don't break the app."""
    if name == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(model=model)
    if name == "openai":
        from .openai_provider import OpenAIProvider

        return OpenAIProvider(model=model)
    if name == "local":
        from .local_provider import LocalProvider

        return LocalProvider(model=model)
    raise ValueError(f"Unknown provider '{name}'. Choose from {PROVIDERS}.")
