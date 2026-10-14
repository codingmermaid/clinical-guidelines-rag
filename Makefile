.PHONY: run test index ingest eval

run:      ## start the API and the web interface
	python start.py

ingest:   ## download (cached) and parse the guidelines, then chunk them
	python -m guideline_rag.ingest
	python -m guideline_rag.chunking

index:    ## rebuild the vector and keyword indexes
	python -m guideline_rag.index --rebuild

test:
	python -m pytest

eval:
	python eval/run_eval.py
