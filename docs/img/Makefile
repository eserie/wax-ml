# See https://stackoverflow.com/questions/38161093/makefile-converting-multiple-dot-files-to-png

TEX_FILES = $(wildcard *.tex)
TEX_PDF=$(TEX_FILES:.tex=.pdf)
TEX_PNG=$(TEX_FILES:.tex=.png)

.PHONY: all
all: pdf clean png

%.pdf: %.tex
	latexmk -f -pdf -silent $<

.PHONY: pdf
pdf: $(TEX_PDF)

%.png: %.tex
	convert -density 300 -quality 90  $(<:.tex=.pdf)  $@

.PHONY: png
png: $(TEX_PNG)

clean:
	# see https://gist.github.com/djsutherland/266983
	latexmk -c
	rm -f *.dvi

