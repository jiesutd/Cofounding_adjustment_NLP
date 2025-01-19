#!/bin/bash

mkdir /path/to/data/mTerms_Cleaned

for mTerms in `find -name "*.csv" | cut -d "." -f2-3`
do
head /path/to/data/mTerms${mTerms} -n 1  | sed 's/ //g' > /path/to/data/mTerms_Cleaned${mTerms}
sed -i 's/leukemia ,/leukemia/g' /path/to/data/mTerms_Cleaned${mTerms} /path/to/data/mTerms_Cleaned${mTerms}
sed '1d; s/ , / /g; s/, / /g; s/ ,/ /g; s/"//g' /path/to/data/mTerms${mTerms} >> /path/to/data/mTerms_Cleaned${mTerms}
sed -i 's/~/_symbol_tilde_/g' /path/to/data/mTerms_Cleaned${mTerms} /path/to/data/mTerms_Cleaned${mTerms}
sed -i 's/@/_symbol_at_/g' /path/to/data/mTerms_Cleaned${mTerms} /path/to/data/mTerms_Cleaned${mTerms}
sed -i 's/#/_symbol_hash_/g' /path/to/data/mTerms_Cleaned${mTerms} /path/to/data/mTerms_Cleaned${mTerms}
sed -i 's/%/_symbol_percent_/g' /path/to/data/mTerms_Cleaned${mTerms} /path/to/data/mTerms_Cleaned${mTerms}
sed -i 's/&/_symbol_and_/g' /path/to/data/mTerms_Cleaned${mTerms} /path/to/data/mTerms_Cleaned${mTerms}
sed -i 's/*/_symbol_asterisk_/g' /path/to/data/mTerms_Cleaned${mTerms} /path/to/data/mTerms_Cleaned${mTerms}
sed -i 's/(/_symbol_open_parenthesis_/g' /path/to/data/mTerms_Cleaned${mTerms} /path/to/data/mTerms_Cleaned${mTerms}
sed -i 's/)/_symbol_close_parenthesis_/g' /path/to/data/mTerms_Cleaned${mTerms} /path/to/data/mTerms_Cleaned${mTerms}
sed -i 's/ - / symbol_hyphen /g' /path/to/data/mTerms_Cleaned${mTerms} /path/to/data/mTerms_Cleaned${mTerms}
sed -i 's/+/_symbol_plus_/g' /path/to/data/mTerms_Cleaned${mTerms} /path/to/data/mTerms_Cleaned${mTerms}
sed -i 's/=/_symbol_equals_/g' /path/to/data/mTerms_Cleaned${mTerms} /path/to/data/mTerms_Cleaned${mTerms}
sed -i 's/\\/_symbol_backslash_/g' /path/to/data/mTerms_Cleaned${mTerms} /path/to/data/mTerms_Cleaned${mTerms}
sed -i 's/?/_symbol_question_mark_/g' /path/to/data/mTerms_Cleaned${mTerms} /path/to/data/mTerms_Cleaned${mTerms}
sed -i 's/\//_symbol_forwardslash_/g' /path/to/data/mTerms_Cleaned${mTerms} /path/to/data/mTerms_Cleaned${mTerms}
sed -i 's/>/_symbol_greater_than_/g' /path/to/data/mTerms_Cleaned${mTerms} /path/to/data/mTerms_Cleaned${mTerms}
sed -i 's/</_symbol_less_than_/g' /path/to/data/mTerms_Cleaned${mTerms} /path/to/data/mTerms_Cleaned${mTerms}
sed -i 's/:/_symbol_colon_/g' /path/to/data/mTerms_Cleaned${mTerms} /path/to/data/mTerms_Cleaned${mTerms}
sed -i 's/;/_symbol_semicolon_/g' /path/to/data/mTerms_Cleaned${mTerms} /path/to/data/mTerms_Cleaned${mTerms}
sed -i "s/'/_symbol_apostrophe_/g" /path/to/data/mTerms_Cleaned${mTerms} /path/to/data/mTerms_Cleaned${mTerms}
sed -i 's/"/_symbol_quotation_/g' /path/to/data/mTerms_Cleaned${mTerms} /path/to/data/mTerms_Cleaned${mTerms}
sed -i 's/ . /_symbol_period_/g' /path/to/data/mTerms_Cleaned${mTerms} /path/to/data/mTerms_Cleaned${mTerms}
sed -i 's/ _symbol_period_,/_symbol_period,/g' /path/to/data/mTerms_Cleaned${mTerms} /path/to/data/mTerms_Cleaned${mTerms}
sed -i 's/\([[:digit:]]\)\.\([[:digit:]]\)/\1_symbol_decimal_\2/g' /path/to/data/mTerms_Cleaned${mTerms} /path/to/data/mTerms_Cleaned${mTerms}
sed -i 's/[^[:alnum:],.-]/_/g' /path/to/data/mTerms_Cleaned${mTerms} /path/to/data/mTerms_Cleaned${mTerms}
sed -i 's/__/_/g' /path/to/data/mTerms_Cleaned${mTerms} /path/to/data/mTerms_Cleaned${mTerms}
sed -i 's/,,,,100_symbol_percent_taken,,,,,,,/,,,100_symbol_percent_taken,,,,,,,/g' /path/to/data/mTerms_Cleaned${mTerms} /path/to/data/mTerms_Cleaned${mTerms}
done
