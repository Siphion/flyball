"""Grading beyond plain numbers: LaTeX answers and code, for the standard benchmarks.

measure_reasoning_cut.py reads numbers and option letters, which is all the
development sets need. The standard benchmarks need two more kinds of answer:

- **LaTeX** (MATH-500): the boxed answer can be a fraction, a root, an interval, a
  tuple, an expression. Both the answer and the reference are normalised the way
  the MATH benchmark's own evaluation does it (Hendrycks et al., `strip_string`),
  with a few additions for what models write today (`\\text{...}`, `{,}`), and a
  numeric comparison when both sides are numbers. Simple and well known, which
  matters more for a published comparison than catching every equivalent form;
  forms it cannot match count as wrong, for every arm alike;
- **code** (HumanEval+, MBPP+): the reply's last Python block, run against the
  benchmark's own tests in the Docker sandbox (sandbox.py), with no network.
"""

from __future__ import annotations

import re

BOX = "\\boxed{"


def boxed_content(text: str) -> str | None:
    """The content of the last \\boxed{...}, braces matched; None without a box."""
    start = text.rfind(BOX)
    if start < 0:
        return None
    depth = 0
    opening = start + len(BOX) - 1
    for index in range(opening, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[opening + 1:index]
    return None


# -- the MATH benchmark's normalisation (Hendrycks et al., 2021) ------------------


def _fix_fracs(string: str) -> str:
    parts = string.split("\\frac")
    out = parts[0]
    for part in parts[1:]:
        out += "\\frac"
        if part[:1] == "{":
            out += part
            continue
        if len(part) < 2:
            return string
        a, b = part[0], part[1]
        if b != "{":
            out += "{" + a + "}{" + b + "}" + part[2:]
        else:
            out += "{" + a + "}" + b + part[2:]
    return out


def _fix_a_slash_b(string: str) -> str:
    pieces = string.split("/")
    if len(pieces) != 2:
        return string
    try:
        a, b = int(pieces[0]), int(pieces[1])
    except ValueError:
        return string
    return "\\frac{" + str(a) + "}{" + str(b) + "}" if string == f"{a}/{b}" else string


def _remove_right_units(string: str) -> str:
    # "\text{ " only ever introduces units in the MATH data.
    pieces = string.split("\\text{ ")
    return pieces[0] if len(pieces) == 2 else string


def _fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    pieces = string.split("\\sqrt")
    out = pieces[0]
    for piece in pieces[1:]:
        out += ("\\sqrt{" + piece[0] + "}" + piece[1:]) if piece and piece[0] != "{" else "\\sqrt" + piece
    return out


def strip_string(string: str) -> str:
    string = string.replace("\n", "").replace("\\!", "").replace("\\\\", "\\")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "").replace("%", "")
    string = string.replace(" .", " 0.").replace("{.", "{0.")
    if not string:
        return string
    if string[0] == ".":
        string = "0" + string
    sides = string.split("=")
    if len(sides) == 2 and len(sides[0]) <= 2:  # "x = 5" is 5
        string = sides[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    return _fix_a_slash_b(string)


# -- what models write today ----------------------------------------------------------

_WRAPPER = re.compile(r"\\(?:text|textbf|mathrm|mathbf|mbox|operatorname)\{([^{}]*)\}")
_SEPARATOR = re.compile(r"(?<=\d)(?:\{,\}|\\[,;: ])(?=\d{3})")


def normalize_latex(answer: str | None) -> str | None:
    """A boxed answer, or a reference, in the one form both are compared in."""
    if answer is None:
        return None
    text = answer.strip().replace("\\displaystyle", "")
    text = _SEPARATOR.sub("", text)
    previous = None
    while previous != text:  # nested wrappers: \text{\textbf{x}}
        previous, text = text, _WRAPPER.sub(r"\1", text)
    text = text.rstrip(".").strip()
    text = strip_string(text)
    plain = text.replace(",", "") if re.fullmatch(r"-?[\d,]+(?:\.\d+)?", text) else None
    if plain is not None:
        value = float(plain)
        return str(int(value)) if value == int(value) else repr(value)
    return text


def latex_in(text: str) -> str | None:
    """The answer a reply commits to, for a LaTeX-answer case."""
    return normalize_latex(boxed_content(text))


# -- code -----------------------------------------------------------------------------

_CODE_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S)


def code_in(text: str) -> str | None:
    """The last Python block of a reply, the code it commits to."""
    blocks = _CODE_BLOCK.findall(text)
    return blocks[-1] if blocks else None


def program_of(case: dict, code: str) -> str:
    """The program that tests `code` the benchmark's own way."""
    if case.get("test_harness"):
        # EvalPlus. HumanEval+'s harness defines check(candidate), called on the
        # entry point; MBPP+'s calls the function itself. The prompt's imports come
        # first, since a reply may leave them out.
        call = "\n\ncheck({})\n".format(case["entry_point"]) if case.get("entry_point") else "\n"
        return "{}\n\n{}\n\n{}{}".format(case.get("code_prefix", ""), code, case["test_harness"], call)
    setup = case.get("test_setup", "")
    return "{}\n{}\n\n{}\n".format(setup, code, "\n".join(case["tests"]))


def grade_code(pairs: list[tuple[dict, str]]) -> list[bool]:
    """Run each (case, reply) in the sandbox; a reply with no code block fails."""
    from sandbox import NUMPY_IMAGE, run_many

    passed = [False] * len(pairs)
    # EvalPlus tests need numpy (sandbox.py --build); the others run on plain Python.
    for harness in (False, True):
        programs, where = [], []
        for index, (case, reply) in enumerate(pairs):
            code = code_in(reply)
            if code is not None and bool(case.get("test_harness")) == harness:
                where.append(index)
                programs.append(program_of(case, code))
        # EvalPlus's inputs are large: some reference solutions need a minute, and
        # one (MBPP+ 255) needs 4 GB, so the limits are wide enough for them.
        results = run_many(programs, image=NUMPY_IMAGE if harness else "",
                           timeout_s=60 if harness else 10, memory="4g" if harness else "1g")
        for index, result in zip(where, results):
            passed[index] = bool(result["passed"])
    return passed
