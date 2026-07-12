all: build/thesis.pdf


TeXOptions = -lualatex \
			 -interaction=nonstopmode \
			 -halt-on-error \
			 -output-directory=build \
			 -f \
                                                                                
build/thesis.pdf: FORCE | build
	latexmk $(TeXOptions) thesis.tex

preview: FORCE | build
	latexmk $(texoptions) -pvc thesis.tex
	
FORCE:

build:
	mkdir -p build/

clean:
	rm -rf build
