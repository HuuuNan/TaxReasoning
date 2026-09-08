# Data

The benchmark data is **not** bundled in this repository. Place it here before
running anything:

```
data/
  data-validation_set-V1.xlsx    development set (187 questions)
  regulation_content.xlsx        tax regulation corpus
```

### `data-validation_set-V1.xlsx`

One row per question, with columns:

| Column | Contents |
|---|---|
| `Case` | The question |
| `Detail_Solution` | Expert-annotated step-by-step solution |
| `Answer` | Gold answer, e.g. `3240元` |
| `Regulation_Title` | Newline-separated titles of the regulations needed to solve it |

### `regulation_content.xlsx`

One row per regulation document, with columns `ID`, `regulation_title`,
`content`. `regulation_title` is what `Regulation_Title` above refers to;
`content` is the full normalised text of the regulation.

### Splits

The development set is the public split. Gold answers for the 753-question test
set are withheld to limit contamination, and evaluation on it runs through an
online submission platform.

### Checking your copy

Once the files are in place, verify that every regulation cited by a question is
present in the corpus:

```bash
python tools/check_data.py
```

The oracle setting feeds a question's cited regulations to the model, so any
title reported as missing means that question silently gets less context than
the setting intends.
