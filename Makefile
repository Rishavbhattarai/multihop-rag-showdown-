PY := .venv/bin/python

.PHONY: setup data index extract graph eval api ui dev

setup:            ## venv + deps + local models
	python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
	cd frontend && npm install
	ollama pull qwen2.5:7b-instruct && ollama pull nomic-embed-text

data:             ## sample MuSiQue -> data/corpus.jsonl, data/questions.jsonl
	$(PY) -m backend.data.prepare

index:            ## embed paragraphs into Chroma (~15s)
	$(PY) -m backend.vector_rag.index

extract:          ## LLM triple extraction, resumable (~2.5h on an M4)
	caffeinate -i $(PY) -m backend.graph_rag.extract

graph:            ## entity resolution + graph + embeddings (~3 min)
	$(PY) -m backend.graph_rag.build

eval:             ## all 4 configurations on 60 questions -> artifacts/results.json (~25 min)
	caffeinate -i $(PY) -m backend.eval.run_eval
	cp artifacts/results.json docs/results.json

api:
	.venv/bin/uvicorn backend.api:app --port 8000

ui:
	cd frontend && npm run dev

dev:              ## API + UI together
	$(MAKE) -j2 api ui
