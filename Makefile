.PHONY: test-cov

test:
	python -m pytest -v --cov=nmanga --cov-config=.coveragerc --cov-report=xml --cov-report=term-missing tests

coverage:
	python -m coverage report

test-cov: test coverage
