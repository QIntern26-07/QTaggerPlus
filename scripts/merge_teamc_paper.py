"""Merge the QSVM/kernel contributions into a copy of the shared Team C paper.

Never writes to `docs/Team_C/` — that directory is the teammate's deliverable and
stays byte-identical. This reads their v0.2 `.tex`, applies the seven insertions
from `docs/paper/teamC_qsvm_sections.tex` plus the six wording corrections, and
writes a separate merged file.

Two outputs:

  clean      -- what to paste into Overleaf. Every insertion carries a
                `% >>> ADDED [n]` comment so the additions stay findable in the
                source after the fact.
  highlight  -- same content, with added prose shaded and reworded sentences
                coloured, for reviewing what changed at a glance.

Run: uv run python scripts/merge_teamc_paper.py
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

from loguru import logger

BASE = Path("docs/Team_C/TeamC_paper_2026-08-06.tex")
BIB = Path("docs/Team_C/refs_2026-08-06.bib")
FIGS = Path("docs/Team_C/figures")
SECTIONS = Path("docs/paper/teamC_qsvm_sections.tex")
NEW_FIG = Path("docs/paper/figures/fig_bandwidth.pdf")

MARKER = re.compile(
    r"% -{10,}\n% \[(\d)\] (?:PASTE INTO|ADD TO)[^\n]*\n(?:%[^\n]*\n)*% -{10,}\n"
)

# (block id, anchor it is inserted before, one-line description for the comment)
INSERTIONS = [
    ("1", "\\end{itemize}\n\nTwo further hybrid variants",
     "Q0 in the model zoo"),
    ("2", "\\subsection{Data notes}",
     "how the kernel arm's protocol differs"),
    ("3", "\\subsection{Multi-seed confirmation}",
     "Results: QSVM vs classical baselines"),
    ("4", "\\subsection{Multi-seed confirmation}",
     "Results: concentration and bandwidth"),
    ("5", "\\section{Limitations}",
     "Discussion: two failure modes, two levers"),
    ("6", "\\section{Consolidation-week task list (completed)}",
     "Limitations: kernel arm"),
    ("7", "\\bibliographystyle{plainnat}",
     "Conclusion: kernel arm"),
]

# (regex, replacement, label) — the six corrections. Patterns tolerate the
# source's line breaks, so they match the file as written rather than as
# quoted in the corrections note.
CORRECTIONS = [
    (r"under a similar protocol in a concurrent internal study,",
     r"under the same protocol in \\S\\ref{sec:qsvm},",
     "1a related work: name the section, not an external study"),
    (r";\s*\nthat study's finding that a \\emph\{local\} measurement fixes.*?"
     r"to build on next\.",
     ". \\\\S\\\\ref{sec:qsvm-bandwidth} shows that on one of our corpora this "
     "concentration is set by the encoding's input scale rather than by the "
     "measurement, and that rescaling reverses it; whether a locality-based "
     "fix instead helps the variational failure mode is tested in "
     "\\\\S\\\\ref{sec:local}.",
     "1b related work: locality was never the kernel-side fix"),
    (r"This\s*\ntests whether the locality principle that fixes kernel "
     r"concentration in\s*\na companion study transfers to the variational "
     r"failure mode\.",
     "This tests whether measurement locality repairs the variational failure "
     "mode, as the concentration literature suggests it can for globally "
     "measured quantum kernels~\\\\cite{thanasilp2024exponential}.",
     "2 model zoo: attribute locality to Thanasilp et al."),
    (r"The\s*\nlocality principle that fixes kernel concentration in a "
     r"companion study\s*\ntherefore does \\emph\{not\} transfer to this "
     r"variational failure mode\s*\nunder the aggregator tested here",
     "Measurement locality therefore does \\\\emph{not} repair this "
     "variational failure mode under the aggregator tested here",
     "3 local-readout results"),
    (r"the locality principle that repairs kernel concentration in a\s*\n"
     r"companion study does not transfer to this variational failure mode\s*\n"
     r"under",
     "measurement locality does not repair this variational failure mode\nunder",
     "4 discussion: the comparison moves to the new Fifth paragraph"),
    (r"Directly tests whether the locality principle behind a companion\s*\n"
     r"kernel-concentration fix transfers to the variational-circuit failure\s*\n"
     r"mode",
     "Directly tests whether measurement locality repairs the variational\n"
     "analogue of kernel concentration",
     "5 task table, Day 4"),
    (r"Tightens confidence intervals to the precision needed to detect the\s*\n"
     r"seed sensitivity a companion kernel study found on a related\s*\n"
     r"architecture",
     "Tightens confidence intervals to the precision needed to detect\n"
     "seed-level variation",
     "6 task table, Day 3: the kernel arm varies folds, not seeds"),
]

# Formatting repairs to the base document, kept separate from the wording
# corrections because they change no claim. Each one is measured: the log goes
# from 5 "Overfull \hbox ... in alignment" warnings at 123.5pt to none.
FORMAT_FIXES = [
    (r"\\begin\{longtable\}\{@\{\}c p\{0\.27\\textwidth\} "
     r"p\{0\.42\\textwidth\} c c@\{\}\}",
     "\\\\begingroup\\\\footnotesize\n\\\\begin{longtable}{@{}c "
     "p{0.24\\\\textwidth} p{0.36\\\\textwidth} p{0.075\\\\textwidth} c@{}}",
     "task-list longtable overflowed the text block by 123.5pt, clipping the "
     "Status column; narrower p-columns, Priority wraps, footnotesize"),
    # longtable is not a group, so a bare \footnotesize before it leaks into
    # everything after -- it silently shrank the whole Conclusion. The size
    # change has to be closed explicitly. Safe as an unanchored replace: the
    # base document contains exactly one longtable.
    (r"\\end\{longtable\}",
     "\\\\end{longtable}\n\\\\endgroup",
     "close the footnotesize group after the longtable"),
]

HIGHLIGHT_PREAMBLE = r"""
% --- review aids, delete before submission ----------------------------------
\usepackage{framed}
\definecolor{shadecolor}{RGB}{255,247,224}
\definecolor{addedrule}{RGB}{200,120,0}
\definecolor{fixedtext}{RGB}{0,90,160}
\newcommand{\addedtag}{\textcolor{addedrule}{\small\textbf{[ADDED]}}\ }
\newcommand{\fixed}[1]{\textcolor{fixedtext}{#1}}
% ----------------------------------------------------------------------------
"""


def load_blocks() -> dict[str, str]:
    src = SECTIONS.read_text()
    hits = list(MARKER.finditer(src))
    out = {}
    for i, m in enumerate(hits):
        end = hits[i + 1].start() if i + 1 < len(hits) else len(src)
        out[m.group(1)] = src[m.end():end].strip()
    return out


def bib_entry(blocks: dict[str, str]) -> str:
    """Block 8 is a commented-out BibTeX entry; uncomment it."""
    return "\n".join(
        line[2:] if line.startswith("% ") else line[1:] if line.startswith("%")
        else line
        for line in blocks["8"].splitlines()
    )


def _replace_with_comment(text: str, pattern: str, repl: str, tag: str,
                          desc: str, flags: int) -> str:
    """Apply one replacement and leave a comment above it saying what changed.

    The `% was:` line is built from the matched text rather than typed by hand,
    so it can never drift from what was actually replaced. LaTeX discards a
    comment line together with its newline, so inserting one mid-paragraph does
    not add a space or break the paragraph.
    """
    m = re.search(pattern, text, flags)
    if not m:
        raise SystemExit(f"[{tag}] pattern did not match: {desc}")
    was = " ".join(m.group(0).split())
    if len(was) > 96:
        was = was[:93] + "..."
    comment = (f"\n% >>> CHANGED [{tag}] -- {desc}\n"
               f"% was: {was}\n")
    return text[:m.start()] + comment + m.expand(repl) + text[m.end():]


def _shade(body: str, kind: str) -> str:
    """Wrap a block for the highlight build.

    Floats escape a shaded environment, so a table or figure gets a tag in its
    caption instead of a background. Everything else is shaded.
    """
    if kind == "item":
        return body.replace("\\item ", "\\item \\addedtag ", 1)
    if kind == "float":
        body = re.sub(r"\\caption\{", r"\\caption{\\addedtag ", body)
        # shade the prose around the floats, leave the floats themselves alone
        parts = re.split(r"(\\begin\{(?:table|figure)\}.*?\\end\{(?:table|figure)\})",
                         body, flags=re.S)
        out = []
        for part in parts:
            if part.strip().startswith("\\begin{table}") or \
               part.strip().startswith("\\begin{figure}"):
                out.append(part)
            elif part.strip():
                out.append("\\begin{snugshade}\n" + part.strip() + "\n\\end{snugshade}")
        return "\n\n".join(out)
    return "\\begin{snugshade}\n" + body + "\n\\end{snugshade}"


def build(highlight: bool) -> str:
    blocks = load_blocks()
    p = BASE.read_text()

    for key, anchor, desc in INSERTIONS:
        if anchor not in p:
            raise SystemExit(f"anchor for block {key} not found: {anchor!r}")
        body = blocks[key]
        if highlight:
            if body.lstrip().startswith("\\item"):
                kind = "item"
            elif "\\begin{table}" in body or "\\begin{figure}" in body:
                kind = "float"
            else:
                kind = "prose"
            body = _shade(body, kind)
        chunk = (f"% >>> ADDED [{key}] — {desc}\n{body}\n"
                 f"% <<< END ADDED [{key}]\n\n")
        p = p.replace(anchor, chunk + anchor, 1)

    for pattern, repl, label in CORRECTIONS:
        tag, desc = label.split(" ", 1)
        if highlight:
            repl = "\\\\fixed{" + repl + "}"
        p = _replace_with_comment(p, pattern, repl, f"C{tag}", desc, re.S)
        logger.info(f"corrected [C{tag}]: {desc}")

    for i, (pattern, repl, label) in enumerate(FORMAT_FIXES, start=1):
        p = _replace_with_comment(p, pattern, repl, f"F{i}", label, 0)
        logger.info(f"format fix [F{i}]: {label}")

    # Comment lines are excluded: the `% was:` markers quote the original
    # wording verbatim, so a naive scan would flag the record of the fix as an
    # unfixed occurrence.
    body = "\n".join(line for line in p.splitlines()
                     if not line.lstrip().startswith("%"))
    leftover = re.findall(
        r"companion (?:kernel )?study|concurrent internal study", body)
    if leftover:
        raise SystemExit(f"{len(leftover)} unfixed mention(s) remain")

    if highlight:
        p = p.replace("\\begin{document}", HIGHLIGHT_PREAMBLE + "\n\\begin{document}", 1)
        p = p.replace("Draft v0.2 (2026-08-06)",
                      "Draft v0.2 + kernel sections --- review copy, "
                      "additions shaded", 1)
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="docs/paper/teamC_merged")
    ap.add_argument("--build-dir", default=None,
                    help="scratch directory for pdflatex; figures are copied here")
    args = ap.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "teamC_paper_merged.tex").write_text(build(highlight=False))
    (out / "teamC_paper_merged_highlight.tex").write_text(build(highlight=True))
    shutil.copy(BIB, out / BIB.name)
    with (out / BIB.name).open("a") as f:
        f.write("\n" + bib_entry(load_blocks()) + "\n")
    logger.info(f"wrote merged sources to {out}")

    if args.build_dir:
        b = Path(args.build_dir)
        b.mkdir(parents=True, exist_ok=True)
        shutil.copytree(FIGS, b / "figures", dirs_exist_ok=True)
        shutil.copy(NEW_FIG, b / "figures" / NEW_FIG.name)
        for name in ("teamC_paper_merged.tex", "teamC_paper_merged_highlight.tex",
                     BIB.name):
            shutil.copy(out / name, b / name)
        logger.info(f"build tree ready in {b}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
