// STAT 534
// Homework 6, Problems 2 and 3

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>

#include "regmodels.h"
#include "matrices.h"


//tests if two vectors are equal
//we assume that the two vectors are sorted
int sameregression(int lenA1,int* A1,int lenA2,int* A2)
{
  int i;

  if(lenA1!=lenA2)
  {
    return 0;
  }

  //the two vectors have the same length
  //are their elements equal?
  for(i=0;i<lenA1;i++)
  {
     if(A1[i]!=A2[i])
     {
       return 0;
     }
  }

  return 1;
}

//this function adds a new regression with predictors A
//to the list of regressions. Here "regressions" represents
//the head of the list, "lenA" is the number of predictors
//and "lml_mc" is the Monte Carlo marginal likelihood of the regression
//with predictors A
void AddRegression(int nMaxRegs, LPRegression regressions,int lenA,int* A, gsl_matrix* beta, double lml_mc, double lml_la)
{
  int i;
  LPRegression p = regressions;
  LPRegression pnext = p->Next;

  int l; // counter for list length

  // printf("\nMaximum number of regressions: %d", nMaxRegs);

  while(NULL!=pnext)
  {
     //return if we have previously found this regression
     if(sameregression(lenA,A,pnext->lenA,pnext->A))
     {
        return;
     }

     //go to the next element in the list if the current
     //regression has a larger log marginal likelihood than
     //the new regression A
     if(pnext->lml_mc>lml_mc)
     {
        p = pnext;
        pnext = p->Next;
     }
     else //otherwise stop; this is where we insert the new regression
     {
        break;
     }
  }

  //create a new element of the list
  LPRegression newp = new Regression;
  newp->lenA = lenA;
  newp->lml_mc = lml_mc;
  newp->lml_la = lml_la;
  newp->A = new int[lenA];
  newp->beta = new double[lenA+1];

  //copy the predictors
  for(i=0;i<lenA;i++)
  {
    newp->A[i] = A[i];
  }


  // Copy the beta coefficients
  for(i=0;i<(lenA+1);i++) {

    newp->beta[i] = gsl_matrix_get(beta, i, 0);

  }

  //insert the new element in the list
  p->Next = newp;
  newp->Next = pnext;

  for(i=0;i<lenA;i++){

      if(i==0) {
        printf("\n input regression [%d]",A[i]);
      } else{
        printf(", [%d]",A[i]);
      }

  }

  printf("\n");

  // Determine the length of the list of regressions
  p = regressions;
  pnext = p->Next;

  l = 0;
  while(NULL != pnext) {

    l += 1;
    p = pnext;
    pnext = p->Next;

  }

  // Delete the last regressions until only nMaxRegs regressions remain
  if(l > nMaxRegs) {

    for(i=0;i<(l-nMaxRegs);i++) {

      DeleteLastRegression(regressions);

    }

  }

  return;
}

// this function deletes the first element of the list
// with the head "regressions"
// remark that the head is not touched
void DeleteFirstRegression(LPRegression regressions) {

  //this is the first regression
  LPRegression p1 = regressions->Next;

  //if the list does not have any elements, return
  if(NULL==p1) {
     return;
  }

  // This is the second regression
  LPRegression p2 = p1->Next;

  // Make the head point to the second regression
  regressions->Next = p2;

  // Delete the first regression
  delete[] p1->A;
  p1->Next = NULL;
  delete p1;

  // printf("Insert code here to delete the first regression");

  return;
}

//this function deletes all the elements of the list
//with the head "regressions"
//remark that the head is not touched
void DeleteAllRegressions(LPRegression regressions)
{
  //this is the first regression
  LPRegression p = regressions->Next;
  LPRegression pnext;

  while(NULL!=p)
  {
    //save the link to the next element of p
    pnext = p->Next;

    //delete the element specified by p
    //first free the memory of the vector of regressors
    delete[] p->A;
    delete[] p->beta;
    p->Next = NULL;
    delete p;

    //move to the next element
    p = pnext;
  }

  return;
}

//this function deletes the last element of the list
//with the head "regressions"
//again, the head is not touched
void DeleteLastRegression(LPRegression regressions)
{
  //this is the element before the first regression
  LPRegression pprev = regressions;
  //this is the first regression
  LPRegression p = regressions->Next;

  //if the list does not have any elements, return
  if(NULL==p)
  {
     return;
  }

  //the last element of the list is the only
  //element that has the "Next" field equal to NULL
  while(NULL!=p->Next)
  {
    pprev = p;
    p = p->Next;
  }

  //now "p" should give the last element
  //delete it
  delete[] p->A;
  delete[] p->beta;
  p->Next = NULL;
  delete p;

  //now the previous element in the list
  //becomes the last element
  pprev->Next = NULL;

  return;
}

//this function saves the regressions in the list with
//head "regressions" in a file with name "filename"
void SaveRegressions(char* filename,LPRegression regressions)
{
  int i;
  //open the output file
  FILE* out = fopen(filename,"w");

  if(NULL==out)
  {
    printf("Cannot open output file [%s]\n",filename);
    exit(1);
  }

  // Header
  fprintf(out, "index\tlml_mc\t\tlml_la\t\tb0\t\tb1\n");


  //this is the first regression
  LPRegression p = regressions->Next;
  while(NULL!=p)
  {

    //now save the predictors
    for(i=0;i<p->lenA;i++)
    {
       fprintf(out,"%i\t",p->A[i]);
    }

    //print the log marginal likelhoods
    fprintf(out,"%.3f\t\t%.3f",p->lml_mc, p->lml_la);

    // beta coefficients
    for(i=0;i<(p->lenA+1);i++)
    {
       fprintf(out,"\t\t%.3f",p->beta[i]);
    }

    fprintf(out,"\n");

    //go to the next regression
    p = p->Next;
  }

  //close the output file
  fclose(out);

  return;
}
