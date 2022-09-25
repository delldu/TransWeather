all:
	@echo "Help"
	@echo "  make format"
	@echo "  make install"
	@echo "  make uninstall"
	@echo "  make clean"

format:
	@echo
	find image_derain -name "*.py" -exec black -l 120 {} \;

install:
	python setup.py install

uninstall:
	pip uninstall image_derain

clean:
	rm -rf build dist image_derain.egg-info __pycache__
