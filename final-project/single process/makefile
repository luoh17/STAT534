RND_LSTDFLG = -lstdc++ -lm -lgsl -lgslcblas -llapack -llapack_atlas
RND_INCLUDE = -I/usr/include/ -I/usr/include/atlas/
RND_LIB = -L/usr/lib/ -L/usr/lib/atlas/ -L/usr/lib/atlas-base/
RND_OBJS = rnd

all:	${RND_OBJS}
	rm -f *.o

matrices.o: matrices.cpp matrices.h
	gcc -g -c matrices.cpp -o matrices.o ${RND_INCLUDE}

regmodels.o: regmodels.cpp regmodels.h
	gcc -g -c regmodels.cpp -o regmodels.o ${RND_INCLUDE}

bayes.o: bayesFunctions.cpp bayes.h matrices.h regmodels.h
	gcc -g -c bayesFunctions.cpp -o bayes.o ${RND_INCLUDE}

main.o: main.cpp matrices.h regmodels.h
	gcc -g -c main.cpp -o main.o ${RND_INCLUDE}

rnd: main.o matrices.o regmodels.o bayes.o
	gcc main.o bayes.o regmodels.o matrices.o -o rnd ${RND_LIB} ${RND_LSTDFLG}

clean:
	rm -f *.o
	rm -f ${RND_OBJS}