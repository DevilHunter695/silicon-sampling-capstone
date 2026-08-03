"""Prompt templates and verbalisation for demographic conditioning."""

from .templates import (
    format_p0_control,
    format_p1_minimal,
    format_p2_structured,
    format_p3_naturalistic,
    build_prompt,
    get_prompt_template,
)

__all__ = [
    "format_p0_control",
    "format_p1_minimal",
    "format_p2_structured",
    "format_p3_naturalistic",
    "build_prompt",
    "get_prompt_template",
]
