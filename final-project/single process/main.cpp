// Stat 534
// Final Project

#include <stdio.h>
#include <gsl/gsl_rng.h>
#include "matrices.h"
#include "regmodels.h"
#include "bayes.h"

// Adds a regression using predictor index to the LPRegression list regressions
void bayesLogistic(int index, int n, int p, int response, gsl_rng* mystream, LPRegression regressions) {

	int i;
  	double lml_la;
  	double lml_mc;
  	int nMaxRegs = 5;
  	int A[p-1];
  	int lenA = -1;

	  // 534finalprojectdata has 148 rows and 61 columns
  	// The first 60 columns are associated with 60 explanatory variables X,
  	// column 61 (the last column) corresponds with the response binary variable Y
    gsl_matrix* data = gsl_matrix_alloc(n, p);
	  FILE * f = fopen("534finalprojectdata.txt", "r");
	  gsl_matrix_fscanf(f, data);
	  fclose(f);

	  // Initialize predictor and response columns
	  gsl_matrix* x = gsl_matrix_alloc(n, 1);
	  gsl_matrix* y = gsl_matrix_alloc(n, 1);
	  for(i=0;i<n;i++) {

		gsl_matrix_set(x, i, 0, gsl_matrix_get(data, i, index));
		gsl_matrix_set(y, i, 0, gsl_matrix_get(data, i, response));

	}

	// compute beta modes using Newton-Raphson algorithm
	gsl_matrix* betaMode = getcoefNR(n, y, x, 1000);

	// compute posterior means for betas
	gsl_matrix* sampleMeans = getPosteriorMeans(mystream, n, y, x, betaMode, 10000);

	printf(" Sample means:\n");
	for(i=0;i<(sampleMeans->size1);i++) {

		printf("    beta%i = %.3f\n", i, gsl_matrix_get(sampleMeans, i, 0));
	}

	// log marginal likelihood -- LaPlace approximation
	printf("\n Posterior log marginal likelihood P(D) estimates:\n");
	lml_la = getLaplaceApprox(n, y, x, betaMode);
	printf("    Laplace approximation = %.3f \n", lml_la);

	// log marginal likelihood -- Monte Carlo integration
	lml_mc = log(getMC(mystream, n, y, x, 10000));
	printf("    Monte Carlo integration = %.3f \n", lml_mc);

	// Add to linked list
	lenA = 1;
    A[0] = index+1;
    AddRegression(nMaxRegs, regressions,
      lenA, A, sampleMeans, lml_mc, lml_la);

	gsl_matrix_free(x);
	gsl_matrix_free(y);
	gsl_matrix_free(betaMode);
	gsl_matrix_free(sampleMeans);
	gsl_matrix_free(data);

}

// Loads file
int main() {

	int n = 148;
  	int p = 61;
  	int response = 60;
  	int i;

  	char outputfilename[] = "bestregressions.txt";

	// Initialize random number generator
  	const gsl_rng_type* T;
  	gsl_rng* r;
  	gsl_rng_env_setup();
  	T = gsl_rng_default;
  	r = gsl_rng_alloc(T);

  	// head of the list of regressions
  	LPRegression regressions = new Regression;
  	regressions->Next = NULL;

  	// Add regression
  	for(i=0;i<response;i++) {

  		bayesLogistic(i, n, p, response, r, regressions);

  	}

  	SaveRegressions(outputfilename,regressions);

  	DeleteAllRegressions(regressions);

	  gsl_rng_free(r);
  	delete regressions; regressions = NULL;

  	return(1);
}
