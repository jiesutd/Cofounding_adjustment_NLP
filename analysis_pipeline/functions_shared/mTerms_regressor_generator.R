mTerms_regressor_generator <- function(cohort = NULL,
                                       refhort = NULL,
                                       join_by_var = NULL,
                                       mTerms_data_path = NULL,
                                       outputdir = NULL,
                                       summary_only = TRUE,
                                       write_output = FALSE,
                                       direct_load = FALSE) {
  most_frequent <- function(v) {
    uniqv <- unique(v)
    uniqv[which.max(tabulate(match(v, uniqv)))]
  }
  
  SASdate <- function(x) {
    as.integer(round(as.numeric(
      difftime(ymd(x), "1960-01-01", units = "days")
    )))
  }
  
  matchColClasses <- function(df1, df2) {
    sharedColNames <- names(df1)[names(df1) %in% names(df2)]
    sharedColTypes <- sapply(df1[, sharedColNames], class)
    
    for (n in sharedColNames) {
      class(df2[, n]) <- sharedColTypes[n]
      
    }
    
    return(df2)
  }
  
  ####Insert Workflow Boolean Here####
  
  #Data Block
  #Make sure this points to the cleaned mTerms data and not the raw data
  mterms_csvs <-
    paste0(mTerms_data_path, "/", (
      list.files(
        path = mTerms_data_path,
        pattern = "*.csv$",
        recursive = TRUE
      )
    ))
  
  if (!file.exists(paste0(outputdir, "/results"))) {
    dir.create(file.path(outputdir, "results"))
  }
  
  if (!file.exists(paste0(outputdir, "/results/mterms"))) {
    dir.create(file.path(outputdir, "results/mterms"))
  }
  
  
  #grpX and cohortX values will need to be replaced with your own grouping
  #and/or cohort values of interest in order for the funciton to work correctly
  if (str_detect(cohort, "max_")) {
    cohort_mterms <- mterms_csvs[grepl("Medicaid", mterms_csvs)]
    Outputdir1 <- "Medicaid/"
    COIs = str_match(str_match(cohort, "max_\\s*(.*?)\\s*.csv")[,-1],
                     "/\\s*(.*?)\\s*/")[-1]
    if (str_detect(cohort, "cohort1")) {
      cohort_mterms <-
        paste0(cohort_mterms[grepl("grpX_|grpX_", cohort_mterms)])
    } else if (str_detect(cohort, "cohort2")) {
      cohort_mterms <-
        paste0(cohort_mterms[grepl("grpX_|grpX_", cohort_mterms)])
    } else if (str_detect(cohort, "cohort3")) {
      cohort_mterms <-
        paste0(cohort_mterms[grepl("grpX_|grpX_", cohort_mterms)])
    } else if (str_detect(cohort, "cohort4")) {
      #This one has a typo on the group variable
      cohort_mterms <-
        paste0(cohort_mterms[grepl("grpX_|grpX_", cohort_mterms)])
    } else if (str_detect(cohort, "cohort5")) {
      cohort_mterms <- paste0(cohort_mterms[grepl("avr", cohort_mterms)])
    } else if (str_detect(cohort, "cohort6")) {
      cohort_mterms <- paste0(cohort_mterms[grepl("ppi", cohort_mterms)])
    } else if (str_detect(cohort, "cohort7")) {
      cohort_mterms <-
        paste0(cohort_mterms[grepl("cohort7", cohort_mterms)])
    } else{
      cohort_mterms <-
        paste0(cohort_mterms[grepl("grpX_|grpX_", cohort_mterms)])
    }
  } else{
    cohort_mterms <- mterms_csvs[grepl("Medicare", mterms_csvs)]
    Outputdir1 <- "Medicare/"
    COIs = str_match(str_match(cohort, "mcr_\\s*(.*?)\\s*.csv")[,-1],
                     "/\\s*(.*?)\\s*/")[-1]
    if (str_detect(cohort, "cohort1")) {
      cohort_mterms <-
        paste0(cohort_mterms[grepl("grpX_|grpX_", cohort_mterms)])
    } else if (str_detect(cohort, "cohort2")) {
      cohort_mterms <-
        paste0(cohort_mterms[grepl("grpX_|grpX_", cohort_mterms)])
    } else if (str_detect(cohort, "cohort3")) {
      cohort_mterms <-
        paste0(cohort_mterms[grepl("grpX_|grpX_", cohort_mterms)])
    } else if (str_detect(cohort, "cohort4")) {
      cohort_mterms <-
        paste0(cohort_mterms[grepl("grpX_|grpX_", cohort_mterms)])
    } else if (str_detect(cohort, "cohort5")) {
      cohort_mterms <- paste0(cohort_mterms[grepl("avr", cohort_mterms)])
    } else if (str_detect(cohort, "cohort6")) {
      cohort_mterms <- paste0(cohort_mterms[grepl("ppi", cohort_mterms)])
    } else if (str_detect(cohort, "cohort7")) {
      cohort_mterms <-
        paste0(cohort_mterms[grepl("cohort8", cohort_mterms)])
    } else{
      cohort_mterms <-
        paste0(cohort_mterms[grepl("grpX_|grpX_", cohort_mterms)])
    }
  }
  
  if (isTRUE(direct_load)) {
    frequency <-
      read_parquet(
        paste0(
          outputdir,
          '/results/mterms/',
          Outputdir1,
          COIs,
          '/summary_output/',
          COIs,
          '_',
          DatasetOI,
          '_frequency_of_mterms.parquet'
        )
      )
    binary <-
      read_parquet(
        paste0(
          outputdir,
          '/results/mterms/',
          Outputdir1,
          COIs,
          '/summary_output/',
          COIs,
          '_',
          DatasetOI,
          '_binary_mterms_columns.parquet'
        )
      )
  } else {
    if (!file.exists(paste0(outputdir, "/results/mterms/", Outputdir1))) {
      dir.create(file.path(outputdir, "results/mterms/", Outputdir1))
    }
    
    if (!file.exists(paste0(outputdir, "/results/mterms/", Outputdir1, "/", COIs))) {
      dir.create(file.path(outputdir, "results/mterms/", Outputdir1, "/", COIs))
    }
    
    if (file.exists(
      paste0(
        outputdir,
        "/results/mterms/",
        DatasetOI,
        "/",
        COIs,
        "/",
        "participant_parquets"
      )
    )) {
      participantout <-
        paste0(
          outputdir,
          "/results/mterms/",
          DatasetOI,
          "/",
          COIs,
          "/",
          "participant_parquets"
        )
    } else {
      dir.create(
        file.path(
          outputdir,
          "results/mterms/",
          DatasetOI,
          "/",
          COIs,
          "/",
          "participant_parquets"
        )
      )
      participantout <-
        paste0(
          outputdir,
          "/results/mterms/",
          DatasetOI,
          "/",
          COIs,
          "/",
          "participant_parquets"
        )
    }
    
    if (file.exists(
      paste0(
        outputdir,
        "/results/mterms/",
        DatasetOI,
        "/",
        COIs,
        "/",
        "summary_output"
      )
    )) {
      summarydir <-
        paste0(outputdir,
               "/results/mterms/",
               DatasetOI,
               "/",
               COIs,
               "/",
               "summary_output")
    } else {
      dir.create(
        file.path(
          outputdir,
          "results/mterms/",
          DatasetOI,
          "/",
          COIs,
          "/",
          "summary_output"
        )
      )
      summarydir <-
        paste0(outputdir,
               "/results/mterms/",
               DatasetOI,
               "/",
               COIs,
               "/",
               "summary_output")
    }
    
    #Here most are manually cast as their data types to avoid conflicts
    temp <-
      open_dataset(
        sources = cohort_mterms,
        format = 'csv',
        unify_schema = TRUE,
        col_types = schema(
          "key" = int64(),
          "noteID" = int64(),
          "expdate" = date32(),
          "notedate" = date32(),
          "diffdays" = int64(),
          "module" = string(),
          "term" = string(),
          "section_text" = string(),
          "negated_mod" = int64(),
          "family_history_mod" = int64(),
          "concept_id" = string(),
          "SNOMED_PrefTerm" = string(),
          "rxcui" = string(),
          "Frequencies" = string(),
          "dispense_amounts" = string(),
          "drug_status" = string(),
          "drug_strenths" = string(),
          "drug_durations" = string(),
          "drug_forms" = string(),
          "drug_intake_times" = string(),
          "drug_routes" = string(),
          "necessities" = string(),
          "refills" = string(),
          "UMLS_CUI" = string(),
          "ProblemCategory" = string(),
          "ICD_Hierarchy" = string()
        )
      )
    
    termsubs <-
      data.frame(
        temp %>%
          select(key, expdate) %>%
          group_by(key, expdate) %>%
          summarize(n = n()) %>%
          collect()
      )
    termsubs$EMPI_Indexdt <-
      as.numeric(paste0(as.character(termsubs$key), as.character(SASdate(
        termsubs$expdate
      ))))
    
    Subs <- refhort$EMPI_Indexdt
    
    if (!isTRUE(summary_only)) {
      tictoc::tic()
      
      for (Subnum in 1:length(refhort$EMPI_Indexdt)) {
        subout <-
          data.frame(refhort %>% filter(EMPI_Indexdt == Subs[Subnum]))
        out <-
          data.frame(temp %>% filter(key == subout$EMPI) %>% collect())
        if ((!Subs[Subnum] %in% termsubs$EMPI_Indexdt) ||
            (!subout$EMPI %in% termsubs$key)) {
          out <- matchColClasses(refhort, out)
          colnames(out)[which(names(out) == "key")] <- "EMPI"
          out <- merge(subout, out, all = TRUE, all.x = TRUE)
          #out$key <- out$EMPI
          out$no_mterms <- 1
          out <-
            out %>% mutate_if(is.character, ~ replace_na(., "NA"))
          out$concept_tags <- "concept_na"
          out$clean_concept_tags <- "concept_na"
          out$term_names <- "term_na"
          out$clean_terms <- "term_na"
          out <-
            dummy_cols(
              out,
              select_columns = "clean_terms",
              remove_selected_columns = FALSE,
              omit_colname_prefix = TRUE
            )
          out <-
            dummy_cols(
              out,
              select_columns = "clean_concept_tags",
              remove_selected_columns = FALSE,
              omit_colname_prefix = TRUE
            )
          out$negated_mod <- as.integer(NA)
          out$family_history_mod <- as.integer(NA)
          out <- matchColClasses(refhort, out)
          write_parquet(
            merge(subout, out),
            paste0(
              outputdir,
              '/results/mterms/',
              Outputdir1,
              COIs,
              '/participant_parquets/',
              as.character(Subs[Subnum]),
              '_factorized_mterms.parquet'
            )
          )
        } else {
          out$no_mterms <- 0
          out <-
            out %>% mutate_if(is.character, ~ replace_na(., "NA"))
          out$EMPI_Indexdt <-
            as.numeric(paste0(as.character(out$key), as.character(SASdate(
              out$expdate
            ))))
          out$concept_tags <- 'NA'
          if (nrow(out[out$module == "ade",]) >  0) {
            out[out$module == "ade",]$concept_tags <-
              paste0("concept_", out[out$module == "ade",]$SNOMED_PrefTerm)
          }
          if (nrow(out[out$module == "Allergy",]) > 0) {
            out[out$module == "Allergy",]$concept_tags <-
              paste0("concept_", out[out$module == "Allergy",]$rxcui)
          }
          if (nrow(out[out$module == "Medication",]) > 0) {
            out[out$module == "Medication",]$concept_tags <-
              paste0("concept_", out[out$module == "Medication",]$concept_id)
          }
          if (nrow(out[out$module == "Problem",]) > 0) {
            out[out$module == "Problem",]$concept_tags <-
              paste0("concept_", out[out$module == "Problem",]$UMLS_CUI)
          }
          if (length(out[out$concept_tags == "concept_", ]$concept_tags) > 0) {
            out[out$concept_tags == "concept_", ]$concept_tags <-
              paste0("concept_NA")
          }
          out$term_names <-
            paste0("term_", out$module, "_", out$term)
          out <- out %>%
            mutate_at(vars(term_names),
                      ~ str_replace(., "\\.", "_symbol_decimal_"))
          out$clean_terms <-
            make_clean_names(out$term_names, allow_dupes = TRUE)
          out$clean_concept_tags <-
            make_clean_names(out$concept_tags, allow_dupes = TRUE)
          out <-
            dummy_cols(
              out,
              select_columns = "clean_terms",
              remove_selected_columns = FALSE,
              omit_colname_prefix = TRUE
            )
          out <-
            dummy_cols(
              out,
              select_columns = "clean_concept_tags",
              remove_selected_columns = FALSE,
              omit_colname_prefix = TRUE
            )
          out <- matchColClasses(refhort, out)
          out$negated_mod <- sapply(out$negated_mod, as.integer)
          out$family_history_mod <-
            sapply(out$family_history_mod, as.integer)
          #This section does the 0/1 inversion when family_history_mod or negated_most_frequent == 1
          out <- out %>%
            mutate(
              across(
                starts_with("term_") &
                  -c("term_names") &
                  -c(colnames(out)[str_detect(colnames(out), "_na$")]) &
                  where(is.numeric),
                ~ if_else(
                  term_names == cur_column() &
                    (family_history_mod | negated_mod) > 0,+!.,
                  .
                )
              ),
              across(
                starts_with("concept_") &
                  -c("concept_tags", "concept_id") &
                  -c(colnames(out)[str_detect(colnames(out), "_na$")]) &
                  where(is.numeric),
                ~ if_else(
                  concept_tags == cur_column() &
                    (family_history_mod | negated_mod) > 0,+!.,
                  .
                )
              )
            )
          write_parquet(
            merge(subout, out),
            paste0(
              outputdir,
              '/results/mterms/',
              Outputdir1,
              COIs,
              '/participant_parquets/',
              as.character(Subs[Subnum]),
              '_factorized_mterms.parquet'
            )
          )
        }
      }
      tictoc::toc()
    }
    tictoc::tic()
    COI_tag <-
      open_dataset(
        sources = paste0(
          outputdir,
          '/results/mterms/',
          Outputdir1,
          COIs,
          '/participant_parquets/'
        ),
        unify_schemas = TRUE,
        format = "parquet"
      )
    
    ## Note: if we want washout of 180 days instead of 365 days, need to change 
    ##       prior365 to prior180 in the terms below
    ## Warning Fla: even if I change to prior180, it doesn't hange output???
    temp <-
      COI_tag |>
      select(
        c(
          -matches("term_|concept_"),
          "clean_terms",
          "term_names",
          "concept_id",
          "concept_tags",
          "clean_concept_tags"
        )
      ) |>
      collect() |>
      mutate_if(is.character, funs(replace_na(., "NA"))) |>
      as.data.frame()
    
    Demographics <-
      temp |>
      group_by(EMPI_Indexdt) |>
      summarise(across(everything(), most_frequent)) |>
      as.data.frame()
    
    Demographics <-
      left_join(Demographics,
                (
                  temp |> filter(module == "ade") |>
                    group_by(EMPI_Indexdt) |>
                    summarise(
                      most_frequent_ade_concept_orig = most_frequent(SNOMED_PrefTerm),
                      most_frequent_ade_derived = most_frequent(clean_concept_tags),
                      concept_ade_module_entries = n()
                    ) |>
                    as.data.frame()
                ))
    
    Demographics <-
      left_join(Demographics,
                (
                  temp |>
                    filter(module == "Allergy") |>
                    group_by(EMPI_Indexdt) |>
                    summarise(
                      most_frequent_allergy_concept_orig = most_frequent(rxcui),
                      most_frequent_allergy_concept_derived = most_frequent(clean_concept_tags),
                      concept_allergy_module_entries = n()
                    ) |>
                    as.data.frame()
                ))
    
    Demographics <-
      left_join(Demographics,
                (
                  temp |> filter(module == "Medication") |>
                    group_by(EMPI_Indexdt) |> summarise(
                      most_frequent_medication_concept_orig = most_frequent(concept_id),
                      most_frequent_medication_concept_derived = most_frequent(clean_concept_tags),
                      concept_medication_module_entries = n()
                    ) |>
                    as.data.frame()
                ))
    
    Demographics <-
      left_join(Demographics,
                (
                  temp |>
                    filter(module == "Problem") |>
                    group_by(EMPI_Indexdt) |>
                    summarise(
                      most_frequent_problem_concept_orig = most_frequent(UMLS_CUI),
                      most_frequent_problem_concept_derived = most_frequent(clean_concept_tags),
                      concept_problem_module_entries = n()
                    ) |>
                    as.data.frame()
                ))
    
    Demographics <-
      left_join(Demographics,
                (
                  temp |> filter(module == "ade") |>
                    group_by(EMPI_Indexdt) |>
                    summarise(
                      most_frequent_ade_term_orig = most_frequent(term),
                      most_frequent_ade_term_derived_unclean = most_frequent(term_names),
                      most_frequent_ade_derived_clean = most_frequent(clean_terms),
                      term_ade_module_entries = n()
                    ) |>
                    as.data.frame()
                ))
    
    Demographics <-
      left_join(Demographics,
                (
                  temp |>
                    filter(module == "Allergy") |>
                    group_by(EMPI_Indexdt) |>
                    summarise(
                      most_frequent_allergy_term_orig = most_frequent(term),
                      most_frequent_allergy_term_derived_unclean = most_frequent(term_names),
                      most_frequent_allergy_term_derived_clean = most_frequent(clean_terms),
                      term_allergy_module_entries = n()
                    ) |>
                    as.data.frame()
                ))
    
    Demographics <-
      left_join(Demographics,
                (
                  temp |> filter(module == "Medication") |>
                    group_by(EMPI_Indexdt) |>
                    summarise(
                      most_frequent_medication_term_orig = most_frequent(term),
                      most_frequent_medication_term_derived_unclean = most_frequent(term_names),
                      most_frequent_medication_term_derived_clean = most_frequent(clean_terms),
                      term_medication_module_entries = n()
                    ) |>
                    as.data.frame()
                ))
    
    Demographics <-
      left_join(Demographics,
                (
                  temp |> filter(module == "Problem") |>
                    group_by(EMPI_Indexdt) |>
                    summarise(
                      most_frequent_problem_term_orig = most_frequent(term),
                      most_frequent_problem_term_derived_unclean = most_frequent(term_names),
                      most_frequent_problem_term_derived_clean = most_frequent(clean_terms),
                      term_problem_module_entries = n()
                    ) |>
                    as.data.frame()
                ))
    
    Demographics <-
      left_join(Demographics,
                (
                  temp |> filter(no_mterms == 0) |>
                    group_by(EMPI_Indexdt) |>
                    summarise(
                      missing_concept = sum(clean_concept_tags == 'concept_na'),
                      missing_term = sum(clean_terms == 'term_na'),
                      total_entries = n()
                    ) |>
                    as.data.frame()
                ))
    
    Demographics <-
      Demographics %>% mutate(
        total_entries = ifelse(is.na(total_entries), 0, total_entries),
        missing_term = ifelse(is.na(missing_term), 1, missing_term)
      )
    
    Demographics <- as_arrow_table(Demographics)
    
    ## Note: if we want washout of 180 days instead of 365 days, need to change 
    ##       prior365 to prior180 in the terms below
    ## Warning Fla: even if I change to prior180, it doesn't hange output???
    frequency <-
      COI_tag |>
      select(where(is.numeric)) |>
      group_by(EMPI_Indexdt) |>
      mutate(across(everything(), ~ coalesce(.x, 0))) |>
      summarise(across(COI_tag$schema$names[grepl("concept_|term_", COI_tag$schema$names)][COI_tag$schema$names[grepl("concept_|term_", COI_tag$schema$names)] %in% c(
        "concept_tags",
        "term_names",
        "clean_terms",
        "clean_concept_tags",
        "concept_id"
      ) == FALSE], sum)) |>
      left_join(Demographics) |>
      compute()
    
    frequency <- data.frame(frequency$to_data_frame())
    
    binary <-
      frequency |> mutate(across(all_of(COI_tag$schema$names[grepl("concept_|term_", COI_tag$schema$names)][COI_tag$schema$names[grepl("concept_|term_", COI_tag$schema$names)] %in% c(
        "concept_tags",
        "term_names",
        "clean_terms",
        "clean_concept_tags",
        "concept_id"
      ) == FALSE]), function(x)
        ifelse(x > 0, 1, x)))
    if (isTRUE(write_output)) {
      write_parquet(
        frequency,
        paste0(
          outputdir,
          '/results/mterms/',
          Outputdir1,
          COIs,
          '/summary_output/',
          COIs,
          '_',
          DatasetOI,
          '_frequency_of_mterms.parquet'
        )
      )
      write_parquet(
        binary,
        paste0(
          outputdir,
          '/results/mterms/',
          Outputdir1,
          COIs,
          '/summary_output/',
          COIs,
          '_',
          DatasetOI,
          '_binary_mterms_columns.parquet'
        )
      )
    }
  }
  
  
  
  
  
  return(list(
    cohort_frequency_mterms = frequency,
    cohort_binarized_mterms = binary
  ))
  
  tictoc::toc()
  
}