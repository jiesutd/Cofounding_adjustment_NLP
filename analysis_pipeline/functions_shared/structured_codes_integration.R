# this function reads in files with empirical covariates used to derive hdps variables

structured_codes_integration <-
  function(cohort = NULL,
           # main cohort for which empirical covariates should be added
           data = paste0(Sys.getenv("path_hdps_r_formatted"), "oa_opioids"),
           dimensions_to_add = c("claims", "rpdr", "combined"),
           join_by_col = "empi_indexdt") {
    # checks ------------------------------------------------------------------
    assertthat::assert_that(!is.null(cohort), msg = "<cohort> not specified")
    assertthat::assert_that(!is.null(data), msg = "<data> not specified")
    assertthat::assert_that(!is.null(dimensions_to_add), msg = "<dimensions_to_add> not specified")
    assertthat::assert_that(!is.null(join_by_col), msg = "<join_by_col> not specified")
    assertthat::assert_that(dimensions_to_add %in% c("claims", "rpdr", "combined"), msg = "<dimensions_to_add> not specified")
    
    
    # select the right files --------------------------------------------------
    #file_names_all <- list.files(path)
    file_names_all <- data
    
    selected_files <- if (dimensions_to_add == "claims") {
      all_mcr_files <- stringr::str_subset(file_names_all, "mcr")
      stringr::str_subset(all_mcr_files, "rpdr", negate = TRUE)
      
    } else if (dimensions_to_add == "rpdr") {
      stringr::str_subset(file_names_all, "rpdr")
      
    } else if (dimensions_to_add == "combined") {
      stringr::str_subset(file_names_all, "mcr|rpdr")
      
    }
    
    # merge covariates to main cohort and binarize -----------------------------------------
    # <path_hdps_r_formatted> contains frequencies of all codes of each dimension
    
    data_frequency <-
      cohort |> mutate(across(as.name(join_by_col), as.character))
    data_binary <-
      cohort |> mutate(across(as.name(join_by_col), as.character))
    
    for (j in 1:length(selected_files)) {
      frequency_out <- arrow::read_csv_arrow(file = selected_files[j])
      
      if (tolower(join_by_col) %in% tolower(colnames(frequency_out))) {
        colnames(frequency_out)[which(tolower(join_by_col) %in%  tolower(colnames(frequency_out)))] <-
          join_by_col
        frequency_out <- frequency_out |>
          mutate(across(as.name(join_by_col), as.character)) |>
          as.data.frame()
      } else {
        print(paste0(
          join_by_col,
          " is not present in ",
          selected_files[j],
          ", exiting."
        ))
        stop()
      }
      
      binary_out <- frequency_out |>
        # binarize frequencies
        dplyr::mutate(dplyr::across(-dplyr::all_of(join_by_col), # all columns but the patient ID column
                                    .fns = ~ if_else(.x > 0, 1, 0, missing = NA))) |>
        dplyr::rename_with(
          .fn = function(.x) {
            paste0(.x, "_hdec")
          },
          .cols = -dplyr::all_of(join_by_col)
        )
      
      # assert that all joined empirical covariates were successfully converted to binary
      assertthat::assert_that(all(apply(binary_out |> dplyr::select(-dplyr::all_of(join_by_col)), 2, function(x) {
        all(x %in% 0:1)
      })),
      msg = "Not all empirical covariates were converted to binary.")
      
      # merge to cohort
      data_frequency <- data_frequency %>%
        dplyr::left_join(frequency_out, by = join_by_col) |>
        dplyr::relocate(dplyr::all_of(join_by_col), 1)
      
      data_binary <- data_binary %>%
        dplyr::left_join(binary_out, by = join_by_col) |>
        dplyr::relocate(dplyr::all_of(join_by_col), 1)
      
    }
    
    # check that no covariate is listed more than one time
    assertthat::assert_that(sum(duplicated(names(data_frequency))) == 0, msg = "For frequency values, at least one column appears more than one time.")
    assertthat::assert_that(sum(duplicated(names(data_binary))) == 0, msg = "For the binarized covariates, at least one covariate appears more than one time.")
    
    # give some information
    message(
      "Data is saved as a list with separate dataframes for frequency and binary data. All returned structured code covariate dimensions which were binarized are labeled with a '_hdec' (high-dimensional empirical covariate) suffix, the frequency data has no such label."
    )
    
    return(
      list(
        cohort_frequency_structured_codes = data_frequency,
        cohort_binarized_structured_codes = data_binary
      )
    )
    
  }