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
	huggingface-cli login --token $(HF) --add-to-git-credential

push-hub:
	huggingface-cli upload JKO9003/Drug-Classification ./App --repo-type=space --commit-message="Sync App files"
	huggingface-cli upload JKO9003/Drug-Classification ./Model /Model --repo-type=space --commit-message="Sync Model"
	huggingface-cli upload JKO9003/Drug-Classification ./Results /Metrics --repo-type=space --commit-message="Sync Metrics"

deploy: hf-login push-hub

all: install format train validate eval update-branch deploy
