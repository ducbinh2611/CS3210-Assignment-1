build:
	gcc -fopenmp sb/sb.c util.c exporter.c goi.c main.c -o goi-parallel.out

clean:
	rm -f *.out *.gch
