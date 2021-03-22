.PHONY: help test build push clean

SERVICE ?=
IMAGE_TAG ?= "local"
export IMAGE_TAG
export SERVICE

ifeq ($(SERVICE), )
	SERVICES = $(shell cat docker-compose*yml | grep "_processor:" | sed 's/\://g' | sort | uniq | xargs)
else
	SERVICES = $(SERVICE)
endif

info:
	@echo "SERVICES:"
	@$(foreach service, $(SERVICES), echo "  * ${service}";)
	@echo "IMAGE_TAG = ${IMAGE_TAG}"

.DEFAULT: help
help:
	@echo "Make Help"
	@echo "make test  - build test image, run tests"
	@echo "make build - build docker image"
	@echo "make push  - push docker image"
	@echo "make clean - remove stale docker images"

test-%:
	@echo "Testing $*"
	IMAGE_TAG=$(IMAGE_TAG) docker-compose build $*
	IMAGE_TAG=$(IMAGE_TAG) docker-compose run $* || { $(MAKE) clean && exit 1; }
	$(MAKE) clean

test: info
	$(foreach service, $(SERVICES), $(MAKE) test-$(service) || exit 1;)

build: info
	IMAGE_TAG=$(IMAGE_TAG) docker-compose -f docker-compose-build.yml build $(SERVICE)

push: info
	IMAGE_TAG=$(IMAGE_TAG) docker-compose -f docker-compose-build.yml push $(SERVICE)

clean:
	docker-compose down
	docker-compose rm