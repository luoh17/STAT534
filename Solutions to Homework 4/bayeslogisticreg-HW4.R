bayesLogistic = function(apredictor,response,data,NumberOfIterations,NumberOfIterationsMC)
{
    #for the function 'mvrnorm'
    require(MASS);
  
    #these are your functions that need to be loaded before they can be called
    source('auxiliaryfunctions-bayeslogisticreg-HW4.R');
 
    #use Newton-Raphson to calculate the mode of the posterior distribution (2)
    #the mode is needed in the Laplace approximation and in the Metropolis-Hastings algorithm
    betaMode = getcoefNR(response,apredictor,data);
    
    #MLEs of beta0 and beta1
    betaMLE = getcoefglm(response,apredictor,data);
    
    #Problem 1: approximate the marginal likelihood (3) using the Laplace approximation
    logmarglik = getLaplaceApprox(response,apredictor,data,betaMode);
    
    #Numerically calculate the marginal likelihood (3) using Monte Carlo (numerically unstable version)
    logmarglikMCraw  = getMonteCarloRaw(response,apredictor,data,NumberOfIterationsMC);
    
    #Numerically calculate the marginal likelihood (3) using Monte Carlo (stable version)
    logmarglikMC  = getMonteCarlo(response,apredictor,data,NumberOfIterationsMC);
    
    #Problem 2: calculate the posterior means of beta0 and beta1
    #by sampling from the joint posterior of beta0 and beta1
    #using the Metropolis-Hastings algorithm
    betaBayes = getPosteriorMeans(response,apredictor,data,betaMode,NumberOfIterations);
    
    return(list(apredictor=apredictor,
                logmarglik=logmarglik,
                logmarglikMCraw=logmarglikMCraw,
                logmarglikMC=logmarglikMC,
                beta0bayes=betaBayes[1],beta1bayes=betaBayes[2],
                beta0mle=betaMLE[1],beta1mle=betaMLE[2]));
}

#PARALLEL VERSION
#datafile = the name of the file with the data
#NumberOfIterations = number of iterations of the Metropolis-Hastings algorithm
#clusterSize = number of separate processes; each process performs one or more
#univariate regressions
main <- function(datafile,NumberOfIterations,NumberOfIterationsMC,clusterSize)
{
  #read the data
  data = read.table(datafile,header=FALSE);
  
  #the sample size is 148 (number of rows)
  #the explanatory variables are the first 60 columns for '534binarydata.txt'
  #the last column is the binary response
  response = ncol(data);
  lastPredictor = ncol(data)-1;
  
  #initialize a cluster for parallel computing
  cluster <- makeCluster(clusterSize, type = "SOCK")
  
  #run the MC3 algorithm from several times
  results = clusterApply(cluster, 1:lastPredictor, bayesLogistic,
                         response,data,NumberOfIterations,NumberOfIterationsMC);
  
  #print out the results
  for(i in 1:lastPredictor)
  {
    cat('Regression of Y on explanatory variable ',results[[i]]$apredictor,
        ' has log marginal likelihood ',results[[i]]$logmarglik,' (Laplace) ',
        results[[i]]$logmarglikMCraw,' (Monte Carlo unstable)',
        results[[i]]$logmarglikMC,' (Monte Carlo stable)',
        ' with beta0 = ',results[[i]]$beta0bayes,' (',results[[i]]$beta0mle,')',
        ' and beta1 = ',results[[i]]$beta1bayes,' (',results[[i]]$beta1mle,')',
        '\n');    
  }
  
  #destroy the cluster
  stopCluster(cluster);  
}

#NOTE: YOU NEED THE PACKAGE 'SNOW' FOR PARALLEL COMPUTING
require(snow);

#this is where the program starts
main('534binarydata.txt',10000,25000,10);