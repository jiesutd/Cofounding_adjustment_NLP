# Cofounding_adjustment_NLP

Natural language processing for scalable feature engineering and ultra-high-dimensional proxy confounding adjustment in healthcare database studies.

This repository lists the source code of our series of work by incorporating natural language processing (NLP) in cofounding adjustment. The main structure of this repository includes two main parts: The general analysis workflow and the nlp feature generation code.

## NLP Feature Generation

- confounding_ngram.py - Generates unigram and bigram features from notes 
- confounding_embeddings.py - Generates word or sentences embeddings 
- confounding_topicmodeling.py - Generates topic modeling features

## The Analysis Workflow
Root Directory:

- .Rprofile - Metadata for how to configure the `R` environment
- nlp_modeling_python.yml - Metadata file containing the `python` packages used and can be used to generate virtual `python` environments
- renv.lock - Metadata file containing the `R` packages used and can be used to configure local `R` environments

functions_shared:

- format_table.R - Helper function to organize result outputs
- mTerms_regressor_generator.R - Function to generate counts and covariates for participant-level mTerms
- read_clustered_sentence_embeddings.R - Function to generate binarized covariates from k-means clustered sentence embedding outputs
- read_clustered_topic_contributions.R - Function to generate raw, max-pooled, and binarized covariates from from k-means clustered topic model outputs
- read_clustered_word_embeddings.R - Function to generate binarized covariates from k-means clustered word embedding outputs
- read_ngrams.R - Function to generate binarized covariates of unigrams and bigrams from ngram outputs
- structured_codes_integration.R - Function to generate covariates for research patient data registry and medical claims data

please refer to [analysis workflow](analysis_pipeline/README.md) for specific details

prediction/scripts:

- 02_modeling_experimental_structured_code_integration_TPD.Rmd - `R` markdown file where each function is separated into code "blocks" that can be individually run like a `jupyter` notebook

workflow_scripts:

- clustering_workflow.py - Script to take processed embeddings output and perform k-means clustering for later processing
- Delimeter_Cleaner_Production.sh - `bash` shell script to replace all symbols and extra commans in mTerms csvs with characters to make them easier to parse and analyze

## Cite

If you use code in this repository in your paper, please cite our papers:

    @article{wyss2025natural,
    title={Natural language processing for scalable feature engineering and ultra-high-dimensional confounding adjustment in healthcare database studies.},
    author={Wyss, Richard and Yang, Jie and Schneeweiss, Sebastian and Plasek, Joseph M and Zhou, Li and Deramus, Thomas and Weberpals, Janick G and Ngan, Kerry and Tsacogianis, Theodore N and Lin, Kueiyu Joshua},
    journal={medRxiv},
    pages={2025--01},
    year={2025},
    publisher={Cold Spring Harbor Laboratory Press}
    }