#Configures .Rprofile and lock.file based on server and R version
if(R.Version()$minor >= "3"){
  Sys.setenv(RENV_PATHS_LOCKFILE = "renv.lock")
  options(repos = c(REPO_NAME = "https://packagemanager.posit.co/cran/__linux__/rhel9/latest"))
  if(system.file(package = "reticulate") != "" | sum(grepl("/nlp-modeling/renv/library/",(.libPaths()))) > 0){
    if(file.exists("renv/activate.R")){
      library(reticulate)
      use_condaenv("/Python/miniforge3/envs/nlp_modeling_python")
      Sys.setenv(python_path=system("which python3", intern = TRUE))
      print("Library paths are:")
      print(.libPaths())
    }
  } else {
    cat("\n\tIf you are seeing this message, then your `renv` has not been activated,\n
        or is not pointing to the correct library path. To fix this, please run:\n
        `renv::activate()`, `renv::init()`, and `renv::load()` as needed until the\n
        path below includes: '/nlp-modeling/renv/library/', or the error disappears.")
    print(.libPaths())
  }
  Sys.setenv(path_data = "/path/to/data/")
  Sys.setenv(path_base_cohorts = "/path/to/data/base_cohorts/")
  Sys.setenv(path_hdps_r_formatted = "/path/to/data/hdps_r_formatted/")
  Sys.setenv(path_hdps_structured = "/path/to/data/hdps_dimensions/")
  Sys.setenv(path_ngrams = "/path/to/data/ngram/")
  Sys.setenv(path_nlp_in = "/path/to/data/embeddings/")
  Sys.setenv(path_mterms_in = "/path/to/data/mTerms/mTerms_Cleaned/")
} else {
  cat("\n\tR version detected does not match with R versions used to construct workflow.\n
        Defaulting to latest versions on CRAN with Posit package manager.\n
        Proceed with caution.")
  options(repos = c(REPO_NAME = "https://packagemanager.posit.co/cran/latest"))
  Sys.setenv(RENV_PATHS_LOCKFILE = "renv_latest.lock")
  cat("\n\tA Python path is not currently configured.\n
        Running Python based workflow is not advised.")
}


if(system.file(package = "fs") != ""){
  library(fs)
  Sys.setenv(TMPDIR = paste0(path_expand("~/tmp")))
  if (!file.exists(Sys.getenv("TMPDIR"))){
    dir.create(file.path(Sys.getenv("TMPDIR")))
  }
}

if (file.exists(paste0(getwd(),"/results"))){
  Sys.setenv(output_dir = paste0(getwd(),"/results"))
  output_dir <- paste0(getwd(),"/results")
} else {
  dir.create(file.path(getwd(), "results"))
  output_dir <- paste0(getwd(),"/results")
  lapply(paste0(output_dir,"/",c("Ngrams","Tabnet","Embeddings")),dir.create)
  Sys.setenv(output_dir = paste0(getwd(),"/results"))
}

if(file.exists("renv/activate.R")){
  source("renv/activate.R")
}
