.PHONY: lint test-cov

lint:
	. .venv/bin/activate; python ci/multilint.py

test:
	. .venv/bin/activate; python -m pytest -v --cov=nmanga --cov-config=.coveragerc --cov-report=xml --cov-report=term-missing tests

coverage:
	. .venv/bin/activate; python -m coverage report

test-cov: test coverage
