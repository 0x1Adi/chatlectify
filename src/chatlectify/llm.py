"""Unified LLM caller. Prefers API key; falls back to local `claude` CLI."""
import os
import shutil
import subprocess


def available(provider: str) -> str | None:
    """Returns 'api', 'cli', or None."""
    if provider == "anthropic":
        if os.environ.get("ANTHROPIC_API_KEY"):
            return "api"
        if shutil.which("claude"):
            return "cli"
    elif provider == "openai":
        if os.environ.get("OPENAI_API_KEY"):
            return "api"
        if shutil.which("codex"):
            return "cli"
    return None


def _claude_cli(prompt: str, system: str | None, model: str | None) -> str:
    args = ["claude", "-p"]
    if system:
        args += ["--append-system-prompt", system]
    if model:
        args += ["--model", model]
    r = subprocess.run(args, input=prompt, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"claude cli failed: {r.stderr.strip()[:300]}")
    return r.stdout.strip()


def _codex_cli(prompt: str, system: str | None, model: str | None) -> str:
    full = f"{system}\n\n{prompt}" if system else prompt
    args = ["codex", "exec", "--skip-git-repo-check"]
    if model:
        args += ["-m", model]
    r = subprocess.run(args, input=full, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"codex cli failed: {r.stderr.strip()[:300]}")
    return r.stdout.strip()


def call(provider: str, prompt: str, system: str | None = None,
         model: str | None = None, max_tokens: int = 4000) -> str:
    mode = available(provider)
    if mode is None:
        raise RuntimeError(
            f"no auth for {provider}: set "
            f"{'ANTHROPIC_API_KEY or install `claude` CLI' if provider == 'anthropic' else 'OPENAI_API_KEY or install `codex` CLI'}"
        )
    if mode == "cli":
        return (_claude_cli if provider == "anthropic" else _codex_cli)(prompt, system, model)
    if provider == "anthropic":
        import anthropic
        kw = {"model": model, "max_tokens": max_tokens,
              "messages": [{"role": "user", "content": prompt}]}
        if system:
            kw["system"] = system
        return anthropic.Anthropic().messages.create(**kw).content[0].text
    import openai
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": prompt}]
    r = openai.OpenAI().chat.completions.create(
        model=model, max_tokens=max_tokens, messages=msgs, temperature=0.7)
    return r.choices[0].message.content
