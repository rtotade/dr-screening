export PYTHONPATH := src

.PHONY: install synthetic smoke train eval serve test docker

install:
	pip install -r requirements.txt

synthetic:
	python scripts/make_synthetic_data.py --patients 200

smoke: synthetic            ## full offline run on synthetic images
	python -m drscreen.train --config configs/smoke.yaml
	python -m drscreen.evaluate --ckpt runs/smoke/best.pt --manifest runs/smoke/val_split.csv --out runs/smoke/eval

train:                      ## real training (edit configs/default.yaml first)
	python -m drscreen.train --config configs/default.yaml

eval:
	python -m drscreen.evaluate --ckpt runs/default/best.pt --manifest data/test.csv --out runs/default/test

serve:
	uvicorn api.main:app --host 0.0.0.0 --port 8000

test:
	pytest -q

docker:
	docker build -t dr-screening .
	docker run -p 8000:8000 -v $$PWD/runs/default:/srv/models -v $$PWD/data:/srv/data dr-screening
