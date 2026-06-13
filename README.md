# Intergenerational Bridge

An AI/NLP web application that transforms modern Turkish text into either historical literary Turkish or the style of an Anatolian folk poet using **RAG + Gemini 2.5 Flash**.

For the detailed Turkish project and presentation guide, see [`PROJE_REHBERI.md`](PROJE_REHBERI.md).

## Setup

Open the project folder in VS Code.

Create a `.env` file in the project root and add your Google AI Studio credentials:

```text
GEMINI_API_KEY=your_google_ai_studio_key
GEMINI_MODEL=gemini-2.5-flash
```

If the Gemini API returns a `429` rate-limit error, wait a few minutes before retrying. You can temporarily use the lighter model by changing the model setting to:

```text
GEMINI_MODEL=gemini-2.5-flash-lite
```

Start the local server:

```bash
python3 -m src.server --port 8001
```

Open the application at:

```text
http://127.0.0.1:8001
```

Use another port if `8001` is already occupied.

## Problem Statement

Modern Turkish differs considerably from historical literary Turkish and Anatolian folk poetry. This project rewrites a modern Turkish input according to a selected literary persona while preserving its original meaning.

Target users include:

- Students exploring historical Turkish literature
- People studying Turkish language and writing styles
- Digital cultural heritage projects

## Datasets

The project uses two Turkish text datasets:

| Dataset | Class / Persona | CSV rows | Usable examples |
|---|---:|---:|---:|
| `genisletilmis_eski_edebi_metinler.csv` | Historical Literary Text | 1,721 | 1,721 |
| `anadolu_ozanlari_veri_seti.csv` | Anatolian Folk Poet | 471 | 386 |
| Total | 2 classes | 2,192 | 2,107 |

CSV columns:

- `yazar_adi` — author name
- `metin_pasaji` — text passage
- `kaynak_link` — source URL
- `donem` — literary period

Very short rows, which are usually titles rather than usable passages, are filtered out.

## Data Preprocessing

The following preprocessing steps are applied:

1. Read the source CSV files.
2. Select `metin_pasaji` as the primary text field.
3. Remove empty and very short passages.
4. Convert text to lowercase.
5. Apply Unicode normalization for Turkish characters.
6. Tokenize the text.
7. Remove common stop words that provide little classification value.
8. Create an approximately 80/20 train-test split for the Naive Bayes evaluation model.

## AI Architecture

The main generation approach is:

```text
RAG + Gemini 2.5 Flash
```

Processing flow:

1. The user enters a modern Turkish text and selects a persona.
2. The input and dataset passages are converted into TF-IDF vectors.
3. Cosine similarity is used to retrieve the most relevant passages for the selected persona.
4. The retrieved passages are inserted into the Gemini prompt as RAG context.
5. Gemini rewrites the input in the target style while preserving its meaning.
6. A Multinomial Naive Bayes model estimates which persona the generated output resembles.

## Why Gemini 2.5 Flash?

This is a text-generation task rather than only a classification task. A classical classifier can identify a style, but it cannot reliably generate natural and varied text.

Gemini 2.5 Flash was selected because it:

- Produces sufficiently strong Turkish text
- Runs in the cloud and does not require a powerful local machine
- Can follow the retrieved examples included in the RAG prompt
- Is fast and practical for an interactive demonstration

## Pretrained Model and Original Work

Gemini 2.5 Flash is used as a pretrained language model. The project does not fine-tune or train Gemini from scratch.

The following components were implemented within the project:

- CSV loading and text cleaning
- TF-IDF vectorization
- Cosine-similarity retrieval
- RAG prompt construction
- Multinomial Naive Bayes evaluation model
- HTTP API and web interface

## Technologies

No third-party Python package is required. The backend uses Python standard-library modules including:

- `csv`
- `json`
- `re`
- `math`
- `os`
- `time`
- `unicodedata`
- `urllib`
- `pathlib`
- `collections`
- `dataclasses`
- `http.server`
- `unittest`

Frontend:

- HTML
- CSS
- Vanilla JavaScript

LLM provider:

- Google Gemini API

## Evaluation Results

Gemini is not trained inside this project and therefore has no project-specific epoch count. The dataset passages are supplied dynamically through RAG.

The project separately trains a Multinomial Naive Bayes classifier to evaluate persona similarity. Its test results are:

| Metric | Result |
|---|---:|
| Accuracy | 98.3% |
| Macro F1 | 97.3% |

Per-class results:

| Class | Precision | Recall | F1 | Test examples |
|---|---:|---:|---:|---:|
| Historical Literary Text | 100.0% | 98.0% | 99.0% | 345 |
| Anatolian Folk Poet | 91.8% | 100.0% | 95.7% | 78 |

These metrics belong to the Naive Bayes classifier, not to Gemini's generation quality.

## Application Features

- Modern Turkish text input
- Historical Literary Text and Anatolian Folk Poet personas
- Adjustable style intensity
- Gemini-powered text transformation
- RAG passage display with similarity scores
- Glossary for historical and literary terms
- Naive Bayes persona prediction
- Dataset and evaluation summaries

## Running Tests

```bash
python3 -m unittest discover -s tests
```

## Project Structure

```text
kusaklararasi-kopru/
  data/source/
    anadolu_ozanlari_veri_seti.csv
    genisletilmis_eski_edebi_metinler.csv
  public/
    assets/
    index.html
    styles.css
    app.js
  src/
    __init__.py
    server.py
    style_engine.py
  tests/
    test_style_engine.py
  .env.example
  PROJE_REHBERI.md
  README.md
```

## Possible Improvements

- Add more authors, periods, and personas
- Replace TF-IDF retrieval with embedding-based semantic search
- Add human evaluation for output quality
- Create a larger parallel dataset for supervised fine-tuning experiments
- Improve retrieval diversity and relevance
