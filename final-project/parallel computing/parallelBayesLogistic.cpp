/*
Stat 534
Final Project

 In parallel, this program computes the log marginal likelihood of various
 univariate Bayes logistic regressions. The output text file contains
 the top 5 regressions sorted by the log marginal likelihood estimated by
 Monte Carlo integration. It also includes the predictor column index j,
 the log marginal likelihood estimated by the Laplace approximation, and the
posterior estimates of the beta coefficients from MCMC by Metropolis-Hastings.

Run the program using the command: mpirun -np 10 parallelBayesLogistic
*/

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <mpi.h>
#include <iomanip>
#include <gsl/gsl_rng.h>
#include "matrices.h"
#include "regmodels.h"
#include "bayes.h"

// MPI communication
#define GETBL	1
#define SHUTDOWNTAG	0

// primary or replica ?
static int myrank;

// Global variables
int n = 148; // Number of observations in dataset
int p = 61; // Number of columns in dataset
int response = 60; // Index of response column
gsl_matrix* data = gsl_matrix_alloc(n, p); // Data matrix
gsl_matrix* y = gsl_matrix_alloc(n, 1); // Response nx1 matrix


// functions
void primary();
void replica(int primaryname);
void bayesLogistic(int index, gsl_rng* mystream, double* out);

int main(int argc, char* argv[])
{
   int i;

   // MPI session begins
   MPI_Init(&argc, &argv);

   // process id
   MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

   // Read in the file
   FILE * f = fopen("534finalprojectdata.txt", "r");
   gsl_matrix_fscanf(f, data);
   fclose(f);

   // Define response variable
   for(i=0;i<n;i++) {

      gsl_matrix_set(y, i, 0, gsl_matrix_get(data, i, response));

   }

   // Branch off to primary or replica function
   // Primary id == 0, replicas' id are in order 1,2,3,...

   if(myrank==0)
   {
      primary();
   }
   else
   {
      replica(myrank);
   }

   gsl_matrix_free(data);
   gsl_matrix_free(y);

   // Finalize MPI
   MPI_Finalize();

   return(1);
}

void primary() {
   int i;
   int rank;
   int ntasks;		// total number of replicas
   int jobsRunning;	// how many replicas are working
   int work[1];		// info to the replicas
   double workresults[5]; // info received from the replicas
   MPI_Status status;	// MPI info

   int nMaxRegs = 5; // Max number of regressions to keep track of
   int A[p-1];
   int lenA = -1; //num of indices
   char outputfilename[] = "bestregressions.txt";

   double lml_la;
   double lml_mc;
   gsl_matrix* betas = gsl_matrix_alloc(2, 1); // Placeholder for coefficient estimates

   //head of the list of regressions
   LPRegression regressions = new Regression;
   regressions->Next = NULL;

   // find out #replicas
   MPI_Comm_size(MPI_COMM_WORLD,&ntasks);

   fprintf(stdout, "Total Number of processors = %d\n",ntasks);

   // Now loop through the variables and compute the R2 values in parallel
   jobsRunning = 1;

   for(i=0;i<response;i++) {
      // This will tell a replica which variable to work on
      work[0] = i;

      if(jobsRunning < ntasks) // Do we have an available processor?
      {
         // Send out a work request
         MPI_Send(&work, 	// the vector with the variable
		            1,
		            MPI_INT,	// vector type
                  jobsRunning,	// the id of the replica to use
                  GETBL,	// tells replica what to do
                  MPI_COMM_WORLD); // send the request out to anyone
				   // who is available
         printf("Primary sends out work request [%d] to replica [%d]\n",
                work[0],jobsRunning);

         // Increase the # of processors in use
         jobsRunning++;

      }
      else // all the processors are in use
      {
         MPI_Recv(workresults,	// where to store the results
 		            5,
		            MPI_DOUBLE,
	 	            MPI_ANY_SOURCE,
		            MPI_ANY_TAG,
		            MPI_COMM_WORLD,
		            &status);     // processor that returned these results

         printf("Primary has received the result of work request [%d] from replica [%d]\n",
                (int) workresults[0],status.MPI_SOURCE);

         // Add the results to the regressions list
         lenA = 1;
         A[0] = (int)workresults[0];
         lml_mc = workresults[1];
         lml_la = workresults[2];
         gsl_matrix_set(betas, 0, 0, workresults[3]);
         gsl_matrix_set(betas, 1, 0, workresults[4]);

         AddRegression(nMaxRegs, regressions,
            lenA, A, betas, lml_mc, lml_la);

         printf("Primary sends out work request [%d] to replica [%d]\n",
                work[0],status.MPI_SOURCE);

         // Send out a new work order to the processors that just
         // returned
         MPI_Send(&work,
                  1,
                  MPI_INT,
                  status.MPI_SOURCE, // the replica that just returned
                  GETBL,
                  MPI_COMM_WORLD);
      } // using all the processors
   } // loop over all the variables


   // collect the work requests that need to be collected
   // loop over all the replicas
   for(rank=1; rank<jobsRunning; rank++)
   {
      MPI_Recv(workresults,
               5,
               MPI_DOUBLE,
               MPI_ANY_SOURCE,	// whoever is ready to report back
               MPI_ANY_TAG,
               MPI_COMM_WORLD,
               &status);

       printf("Primary has received the result of work request [%d]\n",
                (int) workresults[0]);

      //save the results received
      lenA = 1;
      A[0] = (int)workresults[0];
      lml_mc = workresults[1];
      lml_la = workresults[2];
      gsl_matrix_set(betas, 0, 0, workresults[3]);
      gsl_matrix_set(betas, 1, 0, workresults[4]);

      AddRegression(nMaxRegs, regressions,
         lenA, A, betas, lml_mc, lml_la);
   }

   printf("Tell the replicas to shutdown.\n");

   // Shut down the replica processes
   for(rank=1; rank<ntasks; rank++)
   {
      printf("Primary is shutting down replica [%d]\n",rank);
      MPI_Send(0,
	            0,
               MPI_INT,
               rank,		// shutdown this particular node
               SHUTDOWNTAG,		// tell it to shutdown
	       MPI_COMM_WORLD);
   }

   printf("got to the end of Primary code\n");

   //save the list in a file
   SaveRegressions(outputfilename,regressions);

   //delete all regressions
   DeleteAllRegressions(regressions);

   // Free memory
   gsl_matrix_free(betas);
   delete regressions; regressions = NULL;

   // return to the main function
   return;

}

void replica(int replicaname) {
   int work[1];			// the input from primary
   double workresults[5];	// the output for primary
   MPI_Status status;		// for MPI communication

   // Initialize random number generator
   const gsl_rng_type* T;
   gsl_rng* r;
   gsl_rng_env_setup();
   T = gsl_rng_default;
   r = gsl_rng_alloc(T);

   // Set seed based on replica name
   gsl_rng_set(r, replicaname);

   // the replica listens for instructions...
   int notDone = 1;
   while(notDone)
   {
      printf("Replica %d is waiting\n",replicaname);
      MPI_Recv(&work, // the input from primary
	            1,		// the size of the input
	            MPI_INT,		// the type of the input
               0,		// from the PRIMARY node (rank=0)
               MPI_ANY_TAG,	// any type of order is fine
               MPI_COMM_WORLD,
               &status);
      printf("Replica %d just received smth\n",replicaname);



      // switch on the type of work request
      switch(status.MPI_TAG)
      {
         case GETBL:
            // Run the Bayesian logistic regression for this variable
            // ...and save it in the results vector

           printf("Replica %d has received work request [%d]\n",
                  replicaname,work[0]);

	        bayesLogistic(work[0], r, workresults);

            // Send the results
            MPI_Send(&workresults,
                     5,
                     MPI_DOUBLE,
                     0,		// send it to primary
                     0,		// doesn't need a TAG
                     MPI_COMM_WORLD);

            printf("Replica %d finished processing work request [%d]\n",
                   replicaname,work[0]);

            break;

         case SHUTDOWNTAG:
            printf("Replica %d was told to shutdown\n",replicaname);
            return;

         default:
            notDone = 0;
            printf("The replica code should never get here.\n");
            return;
      }
   }

   // Free memory
   gsl_rng_free(r);

   // Return to main function
   return;
}

// Inputs index and random number state and outputs a double vector of length 5
// Output vector includes (in order):
// 0: index of explanatory variable in data,
// 1: log marginal likelihood (Monte Carlo integration)
// 2: log marginal likelihood (Laplace approximation)
// 3: estimated coefficient of beta0 (intercept)
// 4: estimated coefficient of beta1
void bayesLogistic(int index, gsl_rng* mystream, double* out) {

   int i;
   double lml_la;
   double lml_mc;

   // Initialize predictor column
   gsl_matrix* x = gsl_matrix_alloc(n, 1);
   for(i=0;i<n;i++) {

      gsl_matrix_set(x, i, 0, gsl_matrix_get(data, i, index));

   }

   // beta modes using Newton-Raphson algorithm
   gsl_matrix* betaMode = getcoefNR(n, y, x, 1000);
   // printmatrix("betaMode.txt", betaMode);

   // posterior means for betas
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

   // Update output
   out[0] = (double)(index+1);
   out[1] = lml_mc;
   out[2] = lml_la;
   out[3] = gsl_matrix_get(sampleMeans, 0, 0);
   out[4] = gsl_matrix_get(sampleMeans, 1, 0);

   gsl_matrix_free(x);
   gsl_matrix_free(betaMode);
   gsl_matrix_free(sampleMeans);

}
