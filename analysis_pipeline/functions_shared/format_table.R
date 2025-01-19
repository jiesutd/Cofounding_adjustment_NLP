# WRAPPER FOR INSTANTLY NICE LOOKING HTML TABLES --------------------------
format_table <- function(table_in = NULL, # tableone object or final df
                         format_out = "html", # can be of type html or latex
                         show_smd = TRUE,
                         font_size = 16,
                         extra_wide = FALSE,
                         caption = "",
                         ...){
  
  # checks
  if("TableOne" %in% class(table_in)){
    
    table_tmp <- print(
      table_in,
      smd = TRUE,
      printToggle = FALSE,
      ...
    ) %>% 
      dplyr::as_tibble(rownames = "Variable")
    
  }else if("data.frame" %in% class(table_in)){
    
    table_tmp <- as.data.frame(table_in)
    
  }else(
    
    stop("table_in is not a TableOne or data.frame object")
    
  )
  
  add_args <- list(...)
  
  vec_indent <- as.numeric(which(stringr::str_starts(table_tmp$Variable, pattern = " ")==TRUE))
  vec_bold <- as.numeric(which(stringr::str_starts(table_tmp$Variable, pattern = " ", negate = TRUE)==TRUE))
  
  # table call starts
  return_table <- table_tmp %>%
    
    kableExtra::kable(
      booktabs = TRUE,
      linesep = "",
      longtable = TRUE,
      format = "html",
      caption = caption,
      ...
    ) %>%
    
    kableExtra::kable_classic(
      lightable_options = "hover",
      font_size = font_size,
      full_width = extra_wide,
      fixed_thead = TRUE,
      html_font = "Minion") %>% 
    
    # indentation of rows with categorical variables
    kableExtra::add_indent(
      vec_indent, 
      level_of_indent = 1, 
      all_cols = FALSE
      ) %>% 
    
    # table header in bold font
    kableExtra::row_spec(0, bold = TRUE)
  #kableExtra::row_spec(vec_bold, bold = TRUE) %>% 
  #kableExtra::column_spec(1:1, bold = TRUE) 
  
  return(return_table)
  
}
# END