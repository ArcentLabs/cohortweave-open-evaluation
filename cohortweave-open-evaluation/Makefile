.PHONY: install test verify audit reproduce reproduce-real-ct release
install:
	python -m pip install -r requirements.txt

test:
	pytest

verify:
	python scripts/verify_frozen_results.py

audit:
	python scripts/check_public_release.py

reproduce:
	python scripts/reproduce_spark.py

reproduce-real-ct:
	python scripts/reproduce_spark.py --include-real-ct

release:
	python scripts/make_release_archive.py
