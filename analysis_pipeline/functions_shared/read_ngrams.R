# the function looks for .csv files starting with uni or bi
# and return the merged dataset with cohort

read_ngrams <- function(path_ngram_csv = NULL, # path to directory where ngram files live
                        cohort = NULL, # cohort to join on (ID )
                        join_by_var = "<Participant_ID_Variable>"
                        ){
  
  # Make sure join variable is character to avoid surprises
  if(class(cohort[[join_by_var]]) != "character"){
    
    cohort[[join_by_var]] <- as.character(cohort[[join_by_var]])
    
  }
  
  
  # unigrams; read how many unigram files we have
  n_unigram_files <- list.files(path_ngram_csv, pattern = "unigram_[0-9]|unigram_[0-9][0-9].csv")
  n_unigram_file_count <- length(n_unigram_files)
  
  cohort_unigram <- cohort
  
  for(i in 1:n_unigram_file_count){
    
    unigram_i <- data.table::fread(paste0(path_ngram_csv, "unigram_", i, ".csv")) 
    unigram_i[[join_by_var]] <- as.character(unigram_i[[join_by_var]])
    
    # merge to cohort
    cohort_unigram <- cohort_unigram %>% 
      dplyr::left_join(unigram_i, by = join_by_var)
     
  }
  
  # bigrams; read how many bigram files we have
  n_bigram_files <- list.files(path_ngram_csv, pattern = "bigram_[0-9]|bigram_[0-9][0-9].csv")
  n_bigram_file_count <- length(n_bigram_files)
  
  cohort_bigram <- cohort
  
  for(i in 1:n_bigram_file_count){
    
    bigram_i <- data.table::fread(paste0(path_ngram_csv, "bigram_", i, ".csv"))
    bigram_i[[join_by_var]] <- as.character(bigram_i[[join_by_var]])
    
    # merge to cohort
    cohort_bigram <- cohort_bigram %>% 
      dplyr::left_join(bigram_i, by = join_by_var)
    
  }
  
  return(
    list(
      cohort_unigram = cohort_unigram,
      cohort_bigram = cohort_bigram
    )
  )
  
  
}
