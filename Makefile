.PHONY: install test demo audit clean
install: ; pip install -e ".[dev]"
test:    ; pytest -q
demo:    ; advisor run --brain heuristic && advisor run --brain heuristic --sources data/company-week2/sources.yaml
audit:   ; advisor audit
clean:   ; rm -rf out docs/data
