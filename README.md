# What Your Reranker Never Sees

## Contents
1. [Summary](#summary)
2. [Presentation](#presentation)
3. [Architecture](#architecture)
4. [Features](#features)
5. [Prerequisites](#prerequisites)
6. [Installation](#installation)
7. [Usage](#usage)

## Summary <a name="summary"></a>
This is a demo of the chunk_rescorer parameter of the Elastic text similarity reranker retriever. This parameter provides the ability to break down long documents into best-scoring chunks to fit within a reranker's token limits.

## Presentation <a name="presentation"></a>
[Slide deck](https://joeywhelan.github.io/chunk-rescorer/)

## Architecture <a name="architecture"></a>
![architecture](assets/images/arch.png)

## Features <a name="features"></a>
- Jupyter notebook with linear, top-to-bottom execution
- Provisions an Elastic Serverless project via Terraform
- Loads 12 synthetic product manuals into Elastic
- Runs the same queries through Rerank (whole documents) and Chunk Rerank (chunk_rescorer) and compares the results
- Tears down the entire deployment via Terraform

## Prerequisites <a name="prerequisites"></a>
- [`uv`](https://docs.astral.sh/uv/) — Python toolchain (manages Python 3.12)
- [`terraform`](https://developer.hashicorp.com/terraform)
- Elastic Cloud API key (Serverless)

## Installation <a name="installation"></a>
- Install dependencies: `uv sync`
- Copy `terraform/terraform.tfvars.sample` to `terraform/terraform.tfvars` and set your Elastic Cloud API key in the copy

## Usage <a name="usage"></a>
- Launch the notebook: `uv run jupyter lab demo.ipynb`
- Run the cells top to bottom; each section depends on the one before it