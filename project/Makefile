all:
	@echo "Help"
	@echo "  make format"
	@echo "  make install"
	@echo "  make uninstall"
	@echo "  make clean"

format:
	@echo
	find image_weather -name "*.py" -exec black -l 120 {} \;

install:
	python setup.py install

uninstall:
	pip uninstall image_weather

clean:
	rm -rf build dist image_weather.egg-info __pycache__
