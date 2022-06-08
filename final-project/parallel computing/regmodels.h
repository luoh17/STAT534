/*
 FILE: REGMODELS.H

 Linked list
*/


//this avoids including the function headers twice
#ifndef _REGMODELS
#define _REGMODELS

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_matrix.h>

typedef struct myRegression* LPRegression;
typedef struct myRegression Regression;

struct myRegression
{
  int lenA; //number of regressors
  int* A; //regressor index
  double* beta; // regression coefficients
  double lml_mc; // log marginal likelihood of the regression as determined by Monte Carlo integration
  double lml_la; //log marginal likelihood of the regression as determined by Laplace Approximation
  
  LPRegression Next; //link to the next regression
};

void AddRegression(int nMaxRegs, LPRegression regressions,int lenA,int* A, gsl_matrix* beta, double lml_mc, double lml_la);
void DeleteAllRegressions(LPRegression regressions);
void DeleteLastRegression(LPRegression regressions);
void SaveRegressions(char* filename,LPRegression regressions);
void DeleteFirstRegression(LPRegression regressions);

#endif
