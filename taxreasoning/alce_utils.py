"""Text-normalisation helpers vendored from ALCE.

Source: https://github.com/princeton-nlp/ALCE -- ``utils.py``
Copyright (c) 2023 Princeton Natural Language Processing
Licensed under the MIT License; see the NOTICE file at the repository root.

Reproduced verbatim so the citation metrics in ``scripts/eval_citation.py``
match the reference implementation. Only the two functions that script needs
are copied here, to avoid pulling in ALCE's torch/transformers dependencies.
"""

import re
import string


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def remove_citations(sent):
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")
