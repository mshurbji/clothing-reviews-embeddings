
# Clothing Reviews Embeddings

This project builds a simple NLP pipeline that processes clothing reviews using embeddings. It converts text into vector representations, visualizes them, and retrieves similar reviews based on semantic meaning.

---

## What this project does

* Converts review text into embeddings
* Applies dimensionality reduction (t-SNE) for visualization
* Plots 2D representation of reviews
* Finds similar reviews using semantic similarity
* Returns top 3 closest reviews for a given input

---

## Project structure

* `data/` → input dataset (CSV file)
* `src/` → main logic (embeddings, analysis, similarity)
* `output/` → generated results (embeddings, plots)
* `main.py` → runs everything

---

## How to run

Install requirements:

```bash
pip install -r requirements.txt
```

Add your API key in `.env`:

```
OPENAI_API_KEY=your_api_key
```

Run:

```bash
python main.py
```

---

## Notes

* This project was built as part of a practical AI task
* It demonstrates embeddings, dimensionality reduction, and similarity search
* The output includes saved embeddings and visualization results
* Sample data is included for testing
