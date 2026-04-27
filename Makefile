install:
	pip install --upgrade pip && pip install -r requirements.txt

train:
	python train.py

validate:
	python validate.py

report:
	@echo "## MLflow Results" > Results/report.md
	@cat Results/metrics.txt >> Results/report.md

all: install train validate report
