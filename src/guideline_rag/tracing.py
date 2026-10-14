"""Optional local tracing with Arize Phoenix (nothing leaves this computer).

1. Start the Phoenix server in a separate terminal:   python -m phoenix.server.main serve
2. Start the app with tracing on:                      set GUIDELINE_TRACING=1   (PowerShell: $env:GUIDELINE_TRACING=1)
                                                        python start.py
3. Ask a question, then open http://localhost:6006 to see each step (guardrails, retrieval, generation).

When tracing is off, the spans below do nothing and cost nothing.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

from opentelemetry import trace

_tracer = trace.get_tracer("guideline_rag")
_enabled = False


def enable_tracing(endpoint: str = "http://localhost:6006/v1/traces") -> bool:
    """Send traces to a local Phoenix server. Returns True if tracing was switched on."""
    global _enabled
    try:
        from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
        from phoenix.otel import register

        provider = register(project_name="guideline-rag", endpoint=endpoint, set_global_tracer_provider=True)
        LlamaIndexInstrumentor().instrument(tracer_provider=provider)  # traces the embedding calls too
        global _tracer
        _tracer = provider.get_tracer("guideline_rag")
        _enabled = True
    except Exception as err:  # tracing is optional: never stop the app because of it
        print(f"Tracing not enabled: {err!r}")
    return _enabled


def enable_from_env() -> None:
    if os.environ.get("GUIDELINE_TRACING") == "1" and not _enabled:
        enable_tracing()


@contextmanager
def span(name: str, **attributes):
    """Time one pipeline step. Usage: ``with span("retrieval", mode=mode) as s: ... s.set_attribute(...)``."""
    with _tracer.start_as_current_span(name) as s:
        for key, value in attributes.items():
            s.set_attribute(key, str(value))
        yield s
