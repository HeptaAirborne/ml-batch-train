SHELL := /bin/bash

build-release:
	docker build -t ml-batch-train:production .

upload-release:
	docker login -u AWS -p $$(aws ecr get-login-password --region eu-west-1) 349789132088.dkr.ecr.eu-west-1.amazonaws.com
	docker tag ml-batch-train:production 349789132088.dkr.ecr.eu-west-1.amazonaws.com/ml-batch-train:production
	docker push 349789132088.dkr.ecr.eu-west-1.amazonaws.com/ml-batch-train:production


build-staging:
	docker build -t ml-batch-train:staging .

upload-staging:
	docker login -u AWS -p $$(aws ecr get-login-password --region eu-west-1) 349789132088.dkr.ecr.eu-west-1.amazonaws.com
	docker tag ml-batch-train:staging 349789132088.dkr.ecr.eu-west-1.amazonaws.com/ml-batch-train:staging
	docker push 349789132088.dkr.ecr.eu-west-1.amazonaws.com/ml-batch-train:staging


build-playground:
	docker build -t ml-batch-train:playground .

upload-playground:
	docker login -u AWS -p $$(aws ecr get-login-password --region eu-west-1) 349789132088.dkr.ecr.eu-west-1.amazonaws.com
	docker tag ml-batch-train:playground 349789132088.dkr.ecr.eu-west-1.amazonaws.com/ml-batch-train:playground
	docker push 349789132088.dkr.ecr.eu-west-1.amazonaws.com/ml-batch-train:playground