MPI_LSTDFLG = -lstdc++ -lm -lgsl -lgslcblas -llapack -llapack_atlas
MPI_INCLUDE = -I/usr/include/ -I/usr/include/atlas/
MPI_LIB = -L/usr/lib/ -L/usr/lib/atlas/ -L/usr/lib/atlas-base/
MPI_OBJS = parallelBayesLogistic

all:	${MPI_OBJS}
	rm -f *.o

matrices.o: matrices.cpp matrices.h
	gcc -g -c matrices.cpp -o matrices.o ${MPI_INCLUDE}

regmodels.o: regmodels.cpp regmodels.h
	gcc -g -c regmodels.cpp -o regmodels.o ${MPI_INCLUDE}

bayes.o: bayesFunctions.cpp bayes.h matrices.h regmodels.h
	gcc -g -c bayesFunctions.cpp -o bayes.o ${MPI_INCLUDE}	

parallelBayesLogistic.o: parallelBayesLogistic.cpp matrices.h regmodels.h
	mpic++ -g -c parallelBayesLogistic.cpp -o parallelBayesLogistic.o ${MPI_INCLUDE}

parallelBayesLogistic: parallelBayesLogistic.o matrices.o regmodels.o bayes.o
	mpic++ parallelBayesLogistic.o bayes.o regmodels.o matrices.o -o parallelBayesLogistic ${MPI_LIB} ${MPI_LSTDFLG}

clean:
	rm -f *.o
	rm -f ${MPI_OBJS}