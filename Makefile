PROJECT_NAME = ccore
SHELL = /bin/bash

TAR_EXCLUDES = obj *.tar.gz .git tasks .DS_Store .gitignore .testmondata \
		env hpat/mock data .cache .tmontmp __pycache__ \
		*.blg *.aux *.bbl *.bib *.fdb_latexmk *.fls  *.log *.toc *.pyc \
		.ycm_extra_conf.py *.sublime-project *.sublime-workspace hpat*.pdf \
		*.dat latex/*.png latex/*.pdf latex/*.tex *.auxlock *.dpth *.nav \
		*.out *.snm *.vrb
TAR_EXCLUDES_ARG = $(addprefix --exclude=, $(TAR_EXCLUDES))

TAR_TARGET = $(PROJECT_NAME)_b-$(shell git rev-parse --abbrev-ref HEAD)_$(shell git describe --tags --always)
TAR_ARGS = -cvzf

.PRECIOUS: %.o %.d %.g

help:
	@echo "This is just a helper Makefile for easy access to other Makefiles and scripts"
	@echo "Targets: "
	@echo "- pack: Generate a .tar.gz file with the code of the repository, excluding"
	@echo "\tunwanted files."
	@echo "- ccore: Build the C module with the parser."

## Packing

pack:
	@cd ..; tar $(TAR_EXCLUDES_ARG) $(TAR_ARGS) $(TAR_TARGET).tar.gz $(lastword $(notdir $(CURDIR)))
	@mv ../$(TAR_TARGET).tar.gz .
	@echo "Packed $(TAR_TARGET).tar.gz."


ccore:
	cd src && python setup.py install

