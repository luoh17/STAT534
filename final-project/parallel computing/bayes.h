#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <gsl/gsl_matrix.h>

gsl_matrix* makeCholesky(gsl_matrix* K);
void randomMVN(gsl_rng* mystream, gsl_matrix* samples, gsl_matrix* sigma, gsl_matrix* means);
double inverseLogit(double x);
double inverseLogit2(double x);
gsl_matrix* getPi(int n, gsl_matrix* x, gsl_matrix* beta);
gsl_matrix* getPi2(int n, gsl_matrix* x, gsl_matrix* beta);
double logisticLogLikStar(int n, gsl_matrix* y, gsl_matrix* x, gsl_matrix* beta);
double logisticLogLik(int n, gsl_matrix* y, gsl_matrix* x, gsl_matrix* beta);
void getHessian(int n, gsl_matrix* x, gsl_matrix* beta, gsl_matrix* hessian);
void getGradient(int n, gsl_matrix* y, gsl_matrix* x, gsl_matrix* beta, gsl_matrix* gradient);
gsl_matrix* getcoefNR(int n, gsl_matrix* y, gsl_matrix* x, int maxIter = 1000);
gsl_matrix* sampleMH(gsl_rng* mystream, int n,  gsl_matrix* y, gsl_matrix* x, gsl_matrix* betaMode, int niter);
gsl_matrix* getPosteriorMeans(gsl_rng* mystream, int n,  gsl_matrix* y, gsl_matrix* x, gsl_matrix* betaMode, int niter);
double getLaplaceApprox(int n, gsl_matrix* y, gsl_matrix* x, gsl_matrix* betaMode);
double getMC(gsl_rng* mystream, int n, gsl_matrix* y, gsl_matrix* x, int nsamples);