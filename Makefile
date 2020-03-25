.PHONY: clean setup-py install format lint test

clean:
	find . | grep -E '(__pycache__|\.pyc|\.pyo$$)' | xargs rm -rf

setup-py:  # create a setup.py for editable installs
	dephell deps convert

lock:
	poetry lock

install:
	poetry install

format:
	poetry run isort -rc -y .
	poetry run black .

lint:
	poetry run black --check .
	poetry run flake8 .
	poetry run pydocstyle .
	poetry run mypy .

test:
	poetry run pytest tests/
