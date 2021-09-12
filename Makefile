build:
	gcc sb/sb.c util.c exporter.c goi.c main.c -o goi.out

clean:
	rm -f *.out *.gch
