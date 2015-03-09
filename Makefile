all:
	gcc -std=c11 -O3 -o mf mf.c -lm

runC:
	./mf $(file)

runPython:
	python3 mf.py $(file)
