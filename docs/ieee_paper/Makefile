# Makefile for IEEE Paper Compilation
# LegoMCP World-Class Manufacturing System

# Main document
MAIN = main

# LaTeX compiler
LATEX = pdflatex
BIBTEX = bibtex

# Flags
LATEXFLAGS = -interaction=nonstopmode -halt-on-error

# Output directory
OUTDIR = build

.PHONY: all clean view quick

# Full compilation with bibliography
all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib
	@echo "=== First LaTeX pass ==="
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex
	@echo "=== BibTeX pass ==="
	$(BIBTEX) $(MAIN)
	@echo "=== Second LaTeX pass ==="
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex
	@echo "=== Third LaTeX pass ==="
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex
	@echo "=== Build complete ==="
	@echo "Output: $(MAIN).pdf"

# Quick compilation (single pass, no bibliography)
quick:
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex

# View PDF (macOS)
view: $(MAIN).pdf
	open $(MAIN).pdf

# View PDF (Linux)
view-linux: $(MAIN).pdf
	xdg-open $(MAIN).pdf

# Clean auxiliary files
clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz

# Deep clean (including PDF)
distclean: clean
	rm -f $(MAIN).pdf

# Word count (approximate)
wordcount:
	@echo "Approximate word count:"
	@detex $(MAIN).tex | wc -w

# Check for undefined references
check:
	@grep -i "undefined" $(MAIN).log || echo "No undefined references found"
	@grep -i "warning" $(MAIN).log | head -20

# Help
help:
	@echo "LegoMCP IEEE Paper Build System"
	@echo ""
	@echo "Usage:"
	@echo "  make          - Full build with bibliography"
	@echo "  make quick    - Quick build (single pass)"
	@echo "  make view     - Open PDF (macOS)"
	@echo "  make clean    - Remove auxiliary files"
	@echo "  make distclean - Remove all generated files"
	@echo "  make wordcount - Approximate word count"
	@echo "  make check    - Check for warnings/errors"
	@echo ""
	@echo "Requirements:"
	@echo "  - pdflatex"
	@echo "  - bibtex"
	@echo "  - IEEEtran.cls (included in most TeX distributions)"
