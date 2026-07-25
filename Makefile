PYTHON ?= python

.PHONY: install test demo api benchmark holdout

install:
	$(PYTHON) -m pip install -e .

test:
	$(PYTHON) -m compileall researchguard api demo scripts
	$(PYTHON) -m unittest discover -s tests -v

demo:
	$(PYTHON) -m streamlit run demo/app.py

api:
	$(PYTHON) -m uvicorn api.app:app --host 127.0.0.1 --port 8000

benchmark:
	$(PYTHON) scripts/validate_agent_benchmark_v3.py
	$(PYTHON) scripts/compare_planners_v1.py

holdout:
	$(PYTHON) scripts/validate_holdout_v1.py
