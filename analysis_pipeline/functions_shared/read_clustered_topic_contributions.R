read_clustered_topic_contributions <- function(cohort_frame = NULL,
                                               join_by_var = NULL,
                                               input = NULL,
                                               clustsize = NULL) {
  
  if(class(cohort_frame[[join_by_var]]) != "character"){
    
    cohort_frame[[join_by_var]] <- as.character(cohort_frame[[join_by_var]])
    
  }
  
  SASdate <- function(x) {
    round(as.numeric(difftime(x, "1960-01-01", units = "days")))
  }
  
  most_frequent <- function(x) {
    ux <- unique(x)
    ux[which.max(tabulate(match(x, ux)))]
  }
  
  if (!join_by_var %in% names(cohort_frame)) {
    if (tolower(join_by_var) %in% (tolower(names(cohort_frame)))) {
      colnames(cohort_frame)[min(grep(tolower(join_by_var), tolower(names(cohort_frame))))] <-
        join_by_var
    } else{
      stop("Cohort dataframe does not contain joining variable. Aborting.")
    }
  }
  
  cohort_frame[[join_by_var]] <-
    as.character(cohort_frame[[join_by_var]])
  
  pathsplitter = function(x, n, i) {
    do.call(c, lapply(x, function(X)
      paste(unlist(
        strsplit(X, "/")
      )[(n + 1):(i)], collapse = "")))
  }
  
  pathlength = length(str_split_1(input, "/"))
  
  clusterfiles <-
    as.character(dir_ls(
      path = input,
      recurse = TRUE,
      regexp = "[.]parquet$"
    )[str_detect(
      dir_ls(
        path = input,
        recurse = TRUE,
        regexp = "[.]parquet$"
      ),
      paste0(
        ".+(",
        pathsplitter(
          input,
          n = (pathlength - 2),
          i = (pathlength - 1)
        ),
        ").+(_k_",
        clustsize,
        ")"
      )
    )])
  
  
  clustertab <-
    read_parquet(as.character(clusterfiles))
  
   clustersplit <- function(x) {
     cleaner <- str_replace_all(x,c(`\\[` = "", `\\]` = "", `\\(` = "", `\\)`=""))
     len <- length(unlist(strsplit(cleaner,", ")))
     nameout <- paste0("topic_contributions_",str_pad(unlist(strsplit(cleaner,", "))[c(TRUE,FALSE)], 4, pad = "0"))
     varout <- as.numeric(unlist(strsplit(cleaner,", "))[c(FALSE,TRUE)])
     return(list(clustcols = nameout, clustvars = varout))
   }
  
  if (!join_by_var %in% names(clustertab)) {
    if (tolower(join_by_var) %in% (tolower(names(clustertab)))) {
      colnames(clustertab)[min(grep(tolower(join_by_var), tolower(names(clustertab))))] <-
        join_by_var
    } else if (any(tolower(str_split_1(join_by_var, "_")) %in% (tolower(names(clustertab)))) &
               ("expdate" %in% (tolower(names(clustertab))))) {
      VOI <-
        grep(
          tolower(str_split_1(join_by_var, "_"))[tolower(str_split_1(join_by_var, "_")) %in% (tolower(names(clustertab)))],
          names(clustertab),
          ignore.case = TRUE,
          value = TRUE
        )
      clustertab <-
        clustertab |> mutate(
          !!VOI  := as.character(!!sym(VOI)),
          Indexdt = as.character(SASdate(expdate)),
          !!join_by_var := paste0(get(VOI), "", Indexdt)
        )
    } else{
      stop("Topics parquet does not contain joining variable. Aborting.")
    }
  }

   round_to_square <- function(x) {
     str_replace_all(x, pattern = c("\\(" = "[", "\\)" = "]"))
   }
   
   clustertab <- clustertab[!clustertab$topic_contributions == "[]",]
  
   clustout <- clustertab |>
     rowwise() |>
     mutate(topic_contributions = list(fromJSON(round_to_square(topic_contributions),
                                                simplifyMatrix = FALSE
     ))) |>
     ungroup() |>
     mutate(rn = row_number()) |>
     unnest(cols = topic_contributions) |>
     unnest_wider(col = topic_contributions, names_sep = "_") |>
     arrange(topic_contributions_1) |>
     mutate(topic_contributions_1 = paste0("topic_contributions_",topic_contributions_1)) |> 
     pivot_wider(
       id_cols = c(EMPI_Indexdt, rn), names_from = topic_contributions_1,
       values_from = topic_contributions_2,
       values_fill = 0
     ) |>
     arrange(rn)
   
  clust_raw <- clustout

  clustout <- clustout |> 
    group_by(get(join_by_var)) |>
    summarise(across(starts_with("topic_contributions_"), max))
  
  colnames(clustout)[1] <- join_by_var
  
  
  clustout <- left_join(cohort_frame,clustout,by=join_by_var)
  
  clustout <- clustout |> mutate_at(vars(starts_with('topic_contributions_')), list( ~if_else( is.na(.), 0, .) ))
  
  clustbinary <- clustout |> mutate_at(vars(starts_with('topic_contributions_')), ~1 * (. != 0))
  
  write_parquet(clustout, paste0("/path/to/data/topic/results/",DatasetOI,"/",COIs,"/",COIs,"_max_pooled_",clustsize,"_topic_contributions.parquet"))
  write_parquet(clustbinary, paste0("/path/to/data/topic/results/",DatasetOI,"/",COIs,"/",COIs,"_binarized_",clustsize,"_topic_contributions.parquet"))
  write_parquet(clust_raw, paste0("/path/to/data/topic/results/",DatasetOI,"/",COIs,"/",COIs,"_raw_unlisted_",clustsize,"_topic_contributions.parquet"))
  
  return(list(
    cohort_raw_weights = clust_raw,
    cohort_max_topic_contributions = clustout,
    cohort_binarized_topic_contributions = clustbinary
  ))
}