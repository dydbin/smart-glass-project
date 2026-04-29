.PHONY: dev test lint bootstrap smoke-inference-qwen benchmark-qwen-models run-qwen-local-image

bootstrap:
	bash scripts/bootstrap.sh

dev:
	bash scripts/dev.sh

test:
	bash scripts/test.sh

lint:
	bash scripts/lint.sh

smoke-inference-qwen:
	bash scripts/smoke-inference-qwen.sh

benchmark-qwen-models:
	bash scripts/benchmark-qwen-models.sh

run-qwen-local-image:
	bash scripts/run-qwen-local-image.sh "$(IMAGE)"
