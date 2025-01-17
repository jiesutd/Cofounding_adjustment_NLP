# Participant-level Calculations

## Data Folder Structure

For almost all data, the code is assuming that there are is a "Root" data directory, followed by two "Cohort" sub-directories."

As an example for N-grams:
- /root/data/directory/ngrams/
	- /root/data/directory/ngrams/mcr_cohorts/cohort1/actual_data_of_interest_from_mcr_cohort_1*.files
	- /root/data/directory/ngrams/mcr_cohorts/cohort2/actual_data_of_interest_from_mcr_cohort_2*.files
	- /root/data/directory/ngrams/max_cohorts/cohort1/actual_data_of_interest_from_max_cohort_1*.files
	- /root/data/directory/ngrams/max_cohorts/cohort2/actual_data_of_interest_from_max_cohort_2*.files
	
The exception to this is the NLP data, which moves straight into the cohorts:
- /root/data/directory/embeddings/
	- /root/data/directory/embeddings/cohort1
	- /root/data/directory/embeddings/cohort2
	- /root/data/directory/embeddings/cohort3
	- /root/data/directory/embeddings/cohort4

Much of the code is tied to this folder structure, and if should null file, dataframe, or filepath errors occur, look for the following function in the `R` code:

```r
  pathsplitter = function(x, n, i){
    do.call(c, lapply(x, function(X)
      paste(unlist(strsplit(X, "/"))[(n+1):(i)], collapse = "")))
  }
```

Find all instances in the code where it is called and make adjustments to the numeric values. E.g.:

```r
pathsplitter(input, n=(pathlength-2),i=(pathlength-1))
```

And change 2 to 3 or 1 to 2 as necessary for your folder structure.

## Configuring the Python environment

The project is configured and run via `mambaforge` versions:
- mamba 1.5.5
- conda 23.11.0

Once installed, the environment may be created in `mamba` using the command:

`mamba env create -f nlp_modeling_python.yml`

To your desired location.

The default name of the environment is `nlp_modeling_python`.

Before running **ANY** `python` code, the environment must be activated via the command:

`mamba activate nlp_modeling_python`

If this results in an error, check the list of available environments using:

`mamba env list`

And re-run the command with the listed name accordingly.

## _Optional_: Installing R to your mamba environment

If you wish run `R` commands through the terminal instead of through `RStudio`, you may install base `R` to the `mamba` environment using the command:


`mamba install -c conda-forge r-base=4.3.2 renv=1.0.5`


These will be necessary to configure the `R` environment if you wish to run it within your `mamba` environment (which is not advised).


You can optionally install `RStudio` to the environment using:


`mamba install -c conda-forge r-base=4.3.2 rstudio renv=1.0.5`


But this may not work depending on your machine, network, or server configuration and permissions levels.

## Configuring the R environment

**BEFORE RUNNING R**, open the `.Rprofile` file cloned from the repository and replace the following line:

```r
options(repos = c(REPO_NAME = "https://packagemanager.posit.co/cran/__linux__/rhel9/latest"))
```

With the repository corresponding to your own operating system found at the [Posit Package Manager](https://packagemanager.posit.co/client/#/repos/cran/setup) 

In addition, replace the following file paths:

```r
  Sys.setenv(path_data = "/path/to/data/")
  Sys.setenv(path_base_cohorts = "/path/to/data/base_cohorts/")
  Sys.setenv(path_hdps_r_formatted = "/path/to/data/hdps_r_formatted/")
  Sys.setenv(path_hdps_structured = "/path/to/data/hdps_dimensions/")
  Sys.setenv(path_ngrams = "/path/to/data/ngram/")
  Sys.setenv(path_nlp_in = "/path/to/data/embeddings/")
  Sys.setenv(path_mterms_in = "/path/to/data/mTerms/mTerms_Cleaned/")
```

Mapped to each of the following:

- /path/to/data/ - "Root" directory of your data
- /path/to/data/base_cohorts/ - The path to the data of your "cohort" files (e.g. participant IDs with pre-approved covariates, demographic, and/or dosing information) in `.csv` format
- /path/to/data/hdps_r_formatted/ - The path to your files with structured medical codes in a `.csv` format `R` can read (e.g. no unquoted commas for strings)
- /path/to/data/ngram/ - The path to your "raw" structured medical codes
- /path/to/data/embeddings/ - The path to your various sentence and word embedding outputs (by cohort) in `.parquet` format

**IN ADDITION**, open the `renv.lock` file and replace the following line:

```r
"URL": "https://packagemanager.posit.co/cran/latest"
```

With the repository corresponding to your own operating system found at the [Posit Package Manager](https://packagemanager.posit.co/client/#/repos/cran/setup) 

**YOU MAY NOW OPEN `R`** in the terminal or `RStudio` as long you do so in the same directory or change to the directory where **BOTH** the `.RProfile` and `renv.lock` files are contained

Once you have done so, run the following lines of code in `R`:

```r
# If you have not installed renv previously
install.packages("renv")

# If you have installed renv previously
library(renv)

renv::activate()
```

`renv::activate()` should install all necessary packages for the project as long as your package manager is configured to the correct operating system

## Markdown Document Setup

The first two code "blocks" are for data loading and path mapping to cohorts of interest.

For the line that reads `COIs <- 'avr'`, this is just setting `COI` to the string from the cohort in "base_cohorts" you want it to look for (e.g. cohort1)
This name is inherited across outputs and inputs but differences in cases may result in errors for some functions.

DatasetOI <- 'Medicare' is setting the inherited name for ouputs and to set a directory of interest, where:

- Medicare = mcr_cohorts
- Medicaid = max_cohorts

If your folders do not use these strings, you can remove the following lines:

```r
DatasetOI <- 'Medicare'

#Leave these variables alone
if (DatasetOI == 'Medicaid') {
  fval = 'max'
  fdir = 'max_cohorts'
} else if (DatasetOI == 'Medicare') {
  fval = 'mcr'
  fdir = 'mcr_cohorts'
} else{
  fval = ''
}
```

Set `DatasetOI` to an arbitrary string, set `fval` to the pre-pending character of your data directory (if any, e.g. "mcr"), and set fdir to your cohort base directory (if present) above your cohort of interest (e.g. mcr_cohorts)


The value `washout_window = 365` is set to filter the dataset to participants based on a boolean in their own data of whether their medication washout periods were 365 or 180 days. If this does not apply to the data being used, simply set it to an arbitrary or empty string

## Structured Codes

This is performed using the `structured_codes_integration.R` function, which takes four arguments:

```r
structured_codes_integration(
	cohort = The `R` dataframe of the main cohort to which the structured covariates will be added,
	data = A character list of `.csv` files from which the claims, rpdr, or both will be extracted. If left as-is in the `.Rmd`, it should be found outomatically using the `COI` variable,
	dimensions_to_add = Accepts one of the following 3 arguments: "claims", "rpdr", "combined", with the first two adding that specific kind of data and "combined" adding both to the input dataframe,
	join_by_col = The column variable upon which the input dataframe and the claims/rpdr data is to be joined
	)
```

## N-grams

This is performed using the `read_ngrams.R` function, which takes three arguments:

```r
read_ngrams(
  path_ngram_csv = The path to the ngram (unigram and bigram) files in `.csv` format,
  cohort = The `R` dataframe of the main cohort to which the structured covariates will be added,
  join_by_var = The column variable upon which the input dataframe and the claims/rpdr data is to be joined
  )

```

## mTerms

This is performed using the `mTerms_regressor_generator.R` function, which takes eight arguments:

```r
mTerms_regressor_generator(
  cohort = Full path to the "base" cohort file in `.csv` format; which would be the same file read in as the `R` dataframe of the main cohort,
  refhort = The `R` dataframe of the main cohort,
  join_by_var = The column variable upon which the input dataframe and the claims/rpdr data is to be joined,
  mTerms_data_path = Path to where the "mTerms" files in `.csv` format are stored,
  outputdir = Full path to the directory where you would like the `mTerms` outputs to be saved,
  summary_only = TRUE/FALSE boolean indicating whether or not you would like to load the full data (raw + summary) or just the participant summaries,
  write_output = TRUE/FALSE boolean indicating whether or not you would like the outputs generated to be saved to `.parquet` files in the "outputdir",
  direct_load = TRUE/FALSE boolean indicating whether or not you would like previously saved results loaded or intend to generate new results
  )

```

## Sentence Embeddings

This is performed using two functions.

The first is `clustering_workflow.py`, which has three required positional and three flagged arguments

Positional in order of placement:
- Path to the sentence embedding outputs in `.parquet` format
- Level upon which the clustering is supposed to operate, typically "document"
- Column variable upon which the clustering is intended to operate, typically the sentence_embedding_bert column


Flagged arguments:
+ -p/--parquet: The filename(s) structure in regex string format you intend to pull data from for example cohort1_embeddings_sentence_bert_jsl_*.parquet
+ -c/--clusterin: The cluster size you wish to estimate such as 10000 if left blank this will default to a list of [500,1000,2000,5000,10000] and run them all
+ -b/--batch_size: Designates the batch size to use for the clustering default is 20480

```python
mamba activate nlp_modeling_python

python3 clustering_workflow.py /path/to/cohort/files/ document sentence_embedding_bert -p cohort1_embeddings_sentence_bert_jsl_*.parquet -c 10000
```

The second is `read_clustered_sentence_embeddings.R` function, which takes four arguments:

```r
read_clustered_sentence_embeddings(
    cohort_frame = The `R` dataframe of the main cohort,
    join_by_var = The column variable upon which the input dataframe and the claims/rpdr data is to be joined,
    input = Path to the sentence embedding outputs in `.parquet` format,
    clustsize = The cluster size you wish to estimate such as 10000
  )
```

## Word Embeddings

This is performed using two functions.

The first is `clustering_workflow.py`, which has three required positional and three flagged arguments

Positional in order of placement:
- Path to the word embedding outputs in `.parquet` format
- Level upon which the clustering is supposed to operate, typically "token"
- Column variable upon which the clustering is intended to operate, this is dependent upon the model being used either word_embedding_glove or word_embedding_biobert


Flagged arguments:
+ -p/--parquet: The filename(s) structure in regex string format you intend to pull data from for example cohort1_embeddings_word_glove_jsl_*.parquet cohort1_embeddings_word_biobert_jsl_*.parquet
+ -c/--clusterin: The cluster size you wish to estimate such as 10000 if left blank this will default to a list of [500,1000,2000,5000,10000] and run them all
+ -b/--batch_size: Designates the batch size to use for the clustering default is 20480

```python
mamba activate nlp_modeling_python

python3 clustering_workflow.py /path/to/cohort/files/ token word_embedding_biobert -p cohort1_embeddings_word_biobert_jsl_*.parquet -c 10000
```

The second is `read_clustered_word_embeddings.R` function, which takes five arguments:
```r
read_clustered_word_embeddings(
    cohort_frame = The `R` dataframe of the main cohort,
    join_by_var = The column variable upon which the input dataframe and the claims/rpdr data is to be joined,
    input = Path to the word embedding outputs in `.parquet` format,
    model = The model used for the embeddings calculation typically "bert" or "glove",
    clustsize = The cluster size you wish to estimate such as 10000
  )
```

## Topic Models

This is performed using the `read_clustered_topic_contributions.R` function, which takes four arguments:

```r
read_clustered_topic_contributions(
    cohort_frame = The `R` dataframe of the main cohort,
    join_by_var = The column variable upon which the input dataframe and the claims/rpdr data is to be joined,
    input = Path to the topic model outputs in `.parquet` format,
    clustsize = The cluster size of the previously generated outputs typically 1500
  )
```
