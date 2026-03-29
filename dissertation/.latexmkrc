# Latexmk configuration for automatic LaTeX/BibTeX/Biber compilation
# This file ensures proper compilation order and handles dependencies

# Use pdflatex
$pdf_mode = 1;
$postscript_mode = 0;
$dvi_mode = 0;

# Enable shell escape for minted and other packages
$pdflatex = 'pdflatex -shell-escape -interaction=nonstopmode -synctex=0 %O %S';

# Use biber for bibliography
$biber = 'biber %O %B';
$bibtex_use = 2;  # Use biber instead of bibtex

# Ensure proper file dependencies
push @generated_exts, 'bbl', 'fls', 'fdb_latexmk', 'bcf', 'run.xml';

# Clean up auxiliary files when using -c
$clean_ext = 'bbl run.xml fls fdb_latexmk aux bcf blg out';

# Default number of PDF pages to process
$max_repeat = 5;

# File to view
$view = 'pdf';
$pdf_previewer = 'start %S';

# Show warnings in real time
$silent = 0;
