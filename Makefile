USER_NAME ?= github-actions
USER_EMAIL ?= github-actions@github.com

install:
	pip install --upgrade pip && pip install -r requirements.txt

format:
	black *.py || true

train:
	python train.py

validate:
	python validate.py

eval:
	echo "## Model Metrics" > Results/report.md
	cat Results/metrics.txt >> Results/report.md
	cml comment create Results/report.md

update-branch:
	git config --global user.name "$(USER_NAME)"
	git config --global user.email "$(USER_EMAIL)"
	git add Results/ run_id.txt || true
	git diff --staged --quiet || git commit -m "Update with new results"
	git push origin HEAD:main || true

hf-login:
	pip install -U "huggingface_hub[cli]"
	hf auth login --token $(HF)

push-hub:
	hf upload JKO9003/predictor-subastas-casanare ./App . --repo-type=space --commit-message="Sync App files"
	hf upload JKO9003/predictor-subastas-casanare ./Model Model --repo-type=space --commit-message="Sync Model"
	hf upload JKO9003/predictor-subastas-casanare ./Results Metrics --repo-type=space --commit-message="Sync Metrics"

deploy: hf-login push-hub

all: install format train validate eval update-branch deploy
