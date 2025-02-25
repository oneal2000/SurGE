# SurGE

Welcome to the official GitHub repository for SurGE. SurGE is a benchmark and dataset for end-to-end scientific survey generation in the computer science domain. Developed for the SIGIR Resource Track, SurGE provides a comprehensive resource for evaluating automated survey generation systems through both a large-scale dataset and a fully automated evaluation framework.

## Overview

SurGE is designed to push the boundaries of automated survey generation by tackling the complex task of creating coherent, in-depth survey articles from a vast academic literature collection. Unlike traditional IR tasks focused solely on document retrieval, SurGE requires systems to:

- **Retrieve:** Identify relevant academic articles from a corpus of over 1 million papers.
- **Organize:** Construct a structured and hierarchical survey outline.
- **Synthesize:** Generate a coherent narrative with proper citations, reflecting expert-authored surveys.

The benchmark includes 205 carefully curated ground truth surveys, each accompanied by detailed metadata and a corresponding hierarchical structure, along with an extensive literature knowledge base sourced primarily from arXiv.

## Data Release

This repository contains all necessary components for working with the SurGE dataset:

#### Dataset Files & Formats:

- **Ground Truth Surveys:** Each survey includes metadata fields such as title, authors, publication year, abstract, hierarchical structure, and citation lists.
- **Literature Knowledge Base:** A corpus of 1,086,992 academic papers with key fields (e.g., title, authors, abstract, publication date, and category).
- **Auxiliary Mappings:** Topic-to-publication mappings to support systematic survey generation.

The complete dataset can be downloaded from the following link: 

https://drive.google.com/drive/folders/1ZZPeZvjexFcCmgFqxftKeCPn1vYeBR0Q?usp=drive_link

Then you will get the folder `data` .

#### **Ground Truth Survey** 

A **ground truth survey** contains the full content of a survey and its citation information. However, due to space constraints, we cannot display it in its entirety here.

All ground truth surveys are available in `data/surveys.json`

A **survey** consists of the following fields:

| Field        | Description                                               |
| ------------ | --------------------------------------------------------- |
| authors      | List of contributing researchers.                         |
| survey_title | The title of the survey paper.                            |
| year         | The publication year of the survey.                       |
| date         | The exact timestamp of publication.                       |
| category     | Subject classification following the arXiv taxonomy.      |
| abstract     | The abstract of the survey paper.                         |
| structure    | Hierarchical representation of the survey’s organization. |
| survey_id    | A unique identifier for the survey.                       |
| all_cites    | List of document IDs cited in the survey.                 |
| Bertopic_CD  | A diversity measure computed using BERTopic.              |

#### Literature Knowledge Base

The corpus containing all literature articles is available in:  `data/corpus.json`

**Example** : Here, we present how articles are organized in the knowledge base. Overly long abstract has been appropriately shortened.

```
{
        "Title": "Information Geometry of Evolution of Neural Network Parameters While   Training",
        "Authors": [
            "Abhiram Anand Thiruthummal",
            "Eun-jin Kim",
            "Sergiy Shelyag"
        ],
        "Year": "2024",
        "Date": "2024-06-07T23:42:54Z",
        "Abstract": "Artificial neural networks (ANNs) are powerful tools capable of approximating any arbitrary mathematical function, but their interpretability remains limited...",
        "Category": "cs.LG",
        "doc_id": 1086990
    }
```



The following are explanations of each field:

| Key      | Description                                               |
| -------- | --------------------------------------------------------- |
| Title    | The title of the research paper.                          |
| Authors  | A list of contributing researchers.                       |
| Year     | The publication year of the paper.                        |
| Date     | The exact timestamp of the paper’s release.               |
| Abstract | The abstract of the paper.                                |
| Category | The subject classification following the arXiv taxonomy.  |
| doc_id   | A unique identifier assigned for reference and retrieval. |

#### Auxiliary Mappings:

The mapping containing all queries and their corresponding articles is available in: `data/queries.json`

Every topic in topic-to-publication mappings is extracted from the **ground truth survey**. The corresponding articles are the references cited in that paragraph of the original text.

Below is an example, Overly long content has been appropriately shortened.

```
  {
        "original_id": "23870233-7f5b-4ef1-9d38-e6f3adb0fa48",
        "query_id": 486,
        "date": "2020-07-16T09:23:13Z",
        "year": "2020",
        "category": "cs.LG",
        "content": "}\n{\nMachine learning classifiers can perpetuate and amplify the existing systemic injustices in society . Hence, fairness is becoming another important topic. Traditionally...",
        "prefix_titles": [
            [
                "title",
                "Learning from Noisy Labels with Deep Neural Networks: A Survey"
            ],
            [
                "section",
                "Future Research Directions"
            ],
            [
                "subsection",
                "{Robust and Fair Training"
            ]
        ],
        "prefix_titles_query": "What are the future research directions for robust and fair training in the context of learning from noisy labels with deep neural networks?",
        "cites": [
            7771,
            4163,
            3899,
            8740,
            8739
        ],
        "cite_extract_rate": 0.8333333333333334,
        "origin_cites_number": 6
    }
```



The following are explanations of each field:

| key                 | Description                                                  |
| ------------------- | ------------------------------------------------------------ |
| original_id         | The identifier for the section where this query is from.     |
| query_id            | The ID associated with the specific query.                   |
| content             | The content of the section.                                  |
| prefix_titles       | A hierarchical list of titles of the seciton/subsection/paragraph |
| prefix_titles_query | The question this passage is relevant to. The goal of the question is to retrieve relevant documents. |
| cites               | A list of document IDs that are cited within this section.   |
| cite_extract_rate   | The ratio of extracted citations to the total number of citations in the original document. |
| origin_cites_number | The total number of citations originally present in the section. |

## Installation Instructions

### Requirements

Before you begin, make sure you have the following packages installed in your environment:

```
FlagEmbedding
bertopic
safetensors
torch==1.13.1+cu117 
rouge-score
sacrebleu
numpy==1.26.4
openai
transformers==4.44.2
gdown
socksio
```

### Setting Up Your Environment

To create a new environment and install the required packages, follow these steps:

```
conda create -n stard python=3.10
conda activate surge
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

## Evaluation

Each ground truth survey has its own ID. In each folder named of the ID with the corresponding survey output stored for evaluation, you can refer to the example below to try the evaluation. We have provided the articles generated by three baselines on the test set in the paper. For example, with ID baseline as an illustration, the evaluation can be conducted as follows:

```
python src/test_final.py --passage_dir ./baselines/ID/output --save_path ./baselines/ID/output/log.json --device 0 
```

**Arguments Description**

`--passage_dir`  : Directory containing generated survey passages.

`--eval_list` : List of evaluation metrics (space-separated). Default is ALL.

`--survey_path` : Path to the surveys JSON file. Default is `data/surveys.json`.

`--corpus_path` : Path to the corpus JSON file. Default is data/corpus.json.

`--device` : Device ID for computation. Default is 0.

`--api_key` : API key for evaluation services.

`--save_path` : Path to save evaluation results.



As for `eval_list`, we have the choices below:

`ALL` : Evaluate all.

`ROUGE-BLEU` : Evaluate ROUGE and BLEU

`SH-Recall` : Evaluate SH-Recall

`Structure_Quality` : Evaluate Structure_Quality(LLM_as_judge)

`Coverage` : Evaluate Coverage

`Diversity` : Evaluate Diversity

`Relevance-Paper` : Evaluate Relevance-Paper

`Relevance-Section` : Evaluate Relevance-Section

`Relevance-Sentence` : Evaluate Relevance-Sentence

`Logic` : Evaluate Logic
