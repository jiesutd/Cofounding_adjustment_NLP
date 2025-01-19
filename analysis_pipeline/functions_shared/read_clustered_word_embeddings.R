read_clustered_word_embeddings <- function(cohort_frame = NULL,
                                           join_by_var = NULL,
                                           input = NULL,
                                           model = NULL,
                                           clustsize = NULL) {
  cohort_frame[[join_by_var]] <-
    as.character(cohort_frame[[join_by_var]])
  
  pathsplitter = function(x, n, i) {
    do.call(c, lapply(x, function(X)
      paste(unlist(
        strsplit(X, "/")
      )[(n + 1):(i)], collapse = "")))
  }
  
  pathlength = length(str_split_1(input, "/"))
  
  
  if ("reticulate" %in% (.packages())) {
    clusterfiles <-
      dir_ls(path = input,
             recurse = TRUE,
             regexp = "[.]xlsx$")[str_detect(
               dir_ls(
                 path = input,
                 recurse = TRUE,
                 regexp = "[.]xlsx$"
               ),
               paste0(
                 ".+(",
                 pathsplitter(
                   input,
                   n = (pathlength - 2),
                   i = (pathlength - 1)
                 ),
                 ").+(result).+(word_embedding).+(",model,").+(c",
                 clustsize,
                 ")"
               )
             )]
    
    clustertab <-
      read_xlsx(as.character(clusterfiles)) |> filter((!!sym(join_by_var)) %in% unique(cohort_frame[[join_by_var]]))
    
    use_python(python_path)
    np <- import("numpy", convert = FALSE)
    pd <- import("pandas")
    
  } else{
    clusterfiles <-
      dir_ls(path = input,
             recurse = TRUE,
             regexp = "[.]xlsx$")[str_detect(
               dir_ls(
                 path = input,
                 recurse = TRUE,
                 regexp = "[.]xlsx$"
               ),
               paste0(
                 ".+(",
                 pathsplitter(
                   input,
                   n = (pathlength - 2),
                   i = (pathlength - 1)
                 ),
                 ").+(result).+(word_embedding).+(",model,").+(c",
                 clustsize,
                 ")"
               )
             )]
    
    clustertab <-
      read_xlsx(as.character(clusterfiles)) |> filter((!!sym(join_by_var)) %in% unique(cohort_frame[[join_by_var]]))
  }
  
  clustout <-
    as.data.frame(matrix(0, dim(clustertab)[1], clustsize))
  names(clustout) <- paste0("Cluster_ID_", 0:(clustsize - 1))

  clustersplit <- function(x) {
    as.integer(str_split_1(gsub("\\[|]", "", x), ", "))
  }
  
  clustertab <-
    clustertab |> rowwise() |> mutate(string_read = paste(paste0("Cluster_ID_", clustersplit(cluster)), collapse =
                                                            ','))
  
  for (n in 1:dim(clustertab)[1]) {
    clustout[n, ] <-
      (names(clustout) %in% str_split_1(clustertab$string_read[n], pattern = ","))
  }
  
  clustout <- cbind(clustertab[[join_by_var]], clustout)
  names(clustout)[1] <- join_by_var
  
  return(clustout = clustout)
}