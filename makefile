# Declare 'data' as a phony target so it runs even if the folder exists
.PHONY: data

data:
	uv run python -m data.prepare_data