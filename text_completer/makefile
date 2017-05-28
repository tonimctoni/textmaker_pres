all:
	g++ text_completer.cpp -Wall -Wextra -pedantic -std=c++0x -O3 -funroll-loops -ftree-loop-distribution -march=native
run: all
	./a.out
trun: all
	time ./a.out