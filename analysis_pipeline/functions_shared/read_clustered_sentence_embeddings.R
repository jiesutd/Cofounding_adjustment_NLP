read_clustered_sentence_embeddings <- function(cohort_frame = NULL,
                                           join_by_var = NULL,
                                           input = NULL,
                                           clustsize = NULL) {
  
  cohort_frame[[join_by_var]] <- as.character(cohort_frame[[join_by_var]])
  
  pathsplitter = function(x, n, i){
    do.call(c, lapply(x, function(X)
      paste(unlist(strsplit(X, "/"))[(n+1):(i)], collapse = "")))
  }
  
  pathlength = length(str_split_1(input,"/"))
  
  if("reticulate" %in% (.packages())){
    clusterfiles <- as.character(dir_ls(path = input, recurse = TRUE, regexp = "[.]xlsx$")[str_detect(dir_ls(path = input, recurse = TRUE, regexp = "[.]xlsx$"),paste0(".+(", pathsplitter(input, n=(pathlength-2),i=(pathlength-1)),").+(result).+(_sentence_embedding_).+(_c",clustsize,"_)"))])
    
    clustertab <- read_xlsx(as.character(clusterfiles)) |> filter((!!sym(join_by_var)) %in% unique(cohort_frame[[join_by_var]]))
    
    use_python(python_path)
    np <- import("numpy", convert = FALSE)
    pd <- import("pandas")
    
  }else{
    clusterfiles <- as.character(dir_ls(path = input, recurse = TRUE, regexp = "[.]xlsx$")[str_detect(dir_ls(path = input, recurse = TRUE, regexp = "[.]xlsx$"),paste0(".+(", pathsplitter(input, n=(pathlength-2),i=(pathlength-1)),").+(result).+(_sentence_embedding_).+(_c",clustsize,"_)"))])
    
    clustertab <- read_xlsx(as.character(clusterfiles)) |> filter((!!sym(join_by_var)) %in% unique(cohort_frame[[join_by_var]]))
    
  }
    
    clustout <- as.data.frame(matrix(0, dim(clustertab)[1], clustsize))
    names(clustout) <- paste0("Cluster_ID_",0:(clustsize-1))
    #clustout <- cbind(clustertab[[join_by_var]], clustout)
    #names(clustout)[1] <- join_by_var
    
    #stop("The package `reticulate` is currently not attached to your R session.\nIf this package is not available on your server, you should not attempt to run this workflow.\nAborting")
    
    clustersplit <- function(x){
      as.integer(str_split_1(gsub("\\[|]", "", x),", "))
    }
    
    clustertab <- clustertab |> rowwise() |> mutate(string_read = paste(paste0("Cluster_ID_",clustersplit(cluster)), collapse=','))
    
    for(n in 1:dim(clustertab)[1]){
      clustout[n,] <- (names(clustout) %in% str_split_1(clustertab$string_read[n],pattern = ","))
    }
    
    clustout <- cbind(clustertab[[join_by_var]], clustout)
    names(clustout)[1] <- join_by_var
    
    clustout <- clustout |> as_tibble()
    
    return(clustout = clustout)
}