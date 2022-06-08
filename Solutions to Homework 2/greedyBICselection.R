#this function uses 'glm' to fit a logistic regression
#and returns the AIC = deviance + 2*NumberOfCoefficients 
getLogisticAIC <- function(response,explanatory,data)
{
  #check if the regression has no explanatory variables
  if(0==length(explanatory))
  {
     #regression with no explanatory variables
     deviance = glm(data[,response] ~ 1,family=binomial(link=logit))$deviance;
  }
  else
  {
    #regression with at least one explanatory variable
    deviance = glm(data[,response] ~ as.matrix(data[,as.numeric(explanatory)]),family=binomial(link=logit))$deviance;
  }
  #AIC
  return(deviance+2*(1+length(explanatory)));

  #BIC
  #return(deviance+log(nrow(data))*(1+length(explanatory)));
}

#this function checks whether all fitted values are strictly greater
#than 0 and strictly smaller than 1
#this is an empirical way to check if the MLEs exist
isValidLogistic <- function(response,explanatory,data)
{
   epsilon = 1e-20;
  
   if(0==length(explanatory))
   {
      #regression with no explanatory variables
      fittedValues = fitted(glm(data[,response] ~ 1,family=binomial(link=logit)));
   }
   else
   {
     #regression with at least one explanatory variable
     fittedValues = fitted(glm(data[,response] ~ as.matrix(data[,as.numeric(explanatory)]),family=binomial(link=logit)));
   }
   
   if(all(fittedValues>epsilon) && all(fittedValues<1-epsilon))
   {
      return(TRUE); #MLES are well determined 
   }
   return(FALSE); #MLES are not well determined
}

#this is the version of the 'isValidLogistic' function
#based on Charles Geyers RCDD package
isValidLogisticRCDD <- function(response,explanatory,data)
{  
   logisticreg = suppressWarnings(glm(data[,response] ~ as.matrix(data[,as.numeric(explanatory)]),family=binomial(link=logit),x=TRUE));
   tanv = logisticreg$x;
   tanv[data[,response] == 1, ] <- (-tanv[data[,response] == 1, ]);
   vrep = cbind(0, 0, tanv);
   #with exact arithmetic; takes a long time
   #lout = linearity(d2q(vrep), rep = "V");
  
   lout = linearity(vrep, rep = "V");
   return(length(lout)==nrow(data));
}

#Backward greedy search
backwardSearchAIC <- function(response,data,lastPredictor)
{
  #start with the full regression that includes all the variables
  bestRegression = 1:lastPredictor;
  #calculate the AIC of the full regression
  bestRegressionAIC = getLogisticAIC(response,bestRegression,data);
  cat('\n\n\n\nbackwardSearch :: The full logistic regression has AIC = ',bestRegressionAIC,'\n');
  
  #sequentially delete one variable from the current regression
  #and retain that variable that gives the smallest AIC; make the model
  #in which that variable is deleted to be the current model if
  #this leads to a current model with a smaller AIC
  stepNumber = 0;
  while(length(bestRegression)>=1)
  {
    #record the number of steps performed
    stepNumber = stepNumber + 1;
    
    #create a vector that records the AIC values of the regressions
    #we are examining; the number of these regressions is equalt
    #with the number of variables that are in the model
    regAIC = vector('numeric',length(bestRegression));
    
    if(isValidLogisticRCDD(response,bestRegression,data))
    {
      #take each variable that is in the model
      #and delete it from the model
      for(i in 1:length(bestRegression))
      {
        #delete the variable from the model
        newRegression = setdiff(bestRegression,bestRegression[i]);
        #calculate its AIC
        if(isValidLogisticRCDD(response,newRegression,data))
        {
          regAIC[i] = getLogisticAIC(response,newRegression,data);
        }
        else
        {
          regAIC[i] = Inf; 
        }
      }
    }
    else 
    {
      #take each variable that is in the model
      #and delete it from the model
      for(i in 1:length(bestRegression))
      {
        #delete the variable from the model
        newRegression = setdiff(bestRegression,bestRegression[i]);
        #calculate its AIC
        regAIC[i] = getLogisticAIC(response,newRegression,data);
      }
    }
    variableToDelete = bestRegression[which.min(regAIC)];
    cat('\nbackwardSearch :: Step [',stepNumber,'] identified best variable to delete ',variableToDelete);
    cat(' which gives AIC = ',regAIC[which.min(regAIC)],'\n');
    
    #the best variable that could be deleted from the current model
    #is that variable whose deletion from the current model gives
    #the smallest AIC; however, we delete this variable
    #from the current model only if it leads to a smaller AIC
    if((min(regAIC)<bestRegressionAIC)|| !isValidLogisticRCDD(response,bestRegression,data))
    {
       bestRegression = setdiff(bestRegression,variableToDelete);
       bestRegressionAIC = min(regAIC);
       
       cat('backwardSearch :: Current model has AIC = ',bestRegressionAIC,' and contains the variables [');
       if(length(bestRegression)>=1)
       {
        for(i in 1:length(bestRegression)) cat(' ',bestRegression[i]);
       }
       cat(']\n');
    }
    else 
    {
       #we stop if we did not manage to improve the model by including a variable
       cat('\nbackwardSearch :: STOP: could not improve model by deleting a variable\n');
       break;
    }
  }
  
  return(list(aic=bestRegressionAIC,reg=bestRegression));
}

#Forward greedy search with validity check
forwardSearchAICvalid <- function(response,data,lastPredictor)
{
  
  #start with the empty regression with no predictors
  bestRegression = NULL;
  #calculate the AIC of the empty regression
  bestRegressionAIC = getLogisticAIC(response,bestRegression,data);
  cat('\n\n\n\nforwardSearchAICvalid :: The empty logistic regression has AIC = ',bestRegressionAIC,'\n');
  
  #vector that keeps track of all the variables
  #that are currently NOT in the model
  VariablesNotInModel = 1:lastPredictor;
  
  #add variables one at a time to the current regression
  #and retain that variable that gives the smallest values of AIC associated
  #Make the model that includes that variable to be the current model
  #if it gives a smaller AIC than the AIC of the current regression
  
  #stop when there are no variables that can be included in the model
  stepNumber = 0;
  while(length(VariablesNotInModel)>=1)
  {
     #record the number of steps performed
     stepNumber = stepNumber + 1;
    
     #create a vector that records the AIC values of the regressions
     #we are examining; the number of these regressions is equal
     #with the number of variables that are not in the model
     regAIC = vector('numeric',length(VariablesNotInModel));
    
     #take each variable that is not in the model
     #and include it in the model
     for(i in 1:length(VariablesNotInModel))
     {
        #add the variable to the model
        newRegression = c(bestRegression,VariablesNotInModel[i]);
        #calculate its AIC only if the MLEs are well determined
        if(isValidLogisticRCDD(response,newRegression,data))
        {
          regAIC[i] = getLogisticAIC(response,newRegression,data);
        }
        else
        {
          regAIC[i] = Inf; 
        }
     }
     variableToInclude = VariablesNotInModel[which.min(regAIC)];
     cat('\nforwardSearchAICvalid :: Step [',stepNumber,'] identified best variable to add ',variableToInclude);
     cat(' which gives AIC = ',regAIC[which.min(regAIC)],'\n');
     
     #the best variable that could be included in the current model
     #is that variable whose inclusion in the current model gives
     #the smallest AIC; however, we include this variable
     #in the model only if it leads to a smaller AIC
     #than the AIC of the current model
     if(min(regAIC)<bestRegressionAIC)
     {
        bestRegression = sort(c(bestRegression,variableToInclude));
        bestRegressionAIC = min(regAIC);
        #delete that variable from the list of variables that are not in the model
        VariablesNotInModel = setdiff(VariablesNotInModel,variableToInclude);
        
        cat('forwardSearchAICvalid :: Current model has AIC = ',bestRegressionAIC,' and contains the variables [');
        if(length(bestRegression)>=1)
        {
          for(i in 1:length(bestRegression)) cat(' ',bestRegression[i]);
        }
        cat(']\n');
     }
     else
     {
        #we stop if we did not manage to improve the model by including a variable
        cat('\nforwardSearchAICvalid :: STOP: could not improve model by adding a variable\n');
        break;
     }
  }
  
  return(list(aic=bestRegressionAIC,reg=bestRegression));
}

#Forward greedy search
forwardSearchAIC <- function(response,data,lastPredictor)
{
  
  #start with the empty regression with no predictors
  bestRegression = NULL;
  #calculate the AIC of the empty regression
  bestRegressionAIC = getLogisticAIC(response,bestRegression,data);
  cat('\n\n\n\nforwardSearch :: The empty logistic regression has AIC = ',bestRegressionAIC,'\n');
  
  #vector that keeps track of all the variables
  #that are currently NOT in the model
  VariablesNotInModel = 1:lastPredictor;
  
  #add variables one at a time to the current regression
  #and retain that variable that gives the smallest values of AIC associated
  #Make the model that includes that variable to be the current model
  #if it gives a smaller AIC than the AIC of the current regression
  
  #stop when there are no variables that can be included in the model
  stepNumber = 0;
  while(length(VariablesNotInModel)>=1)
  {
     #record the number of steps performed
     stepNumber = stepNumber + 1;
    
     #create a vector that records the AIC values of the regressions
     #we are examining; the number of these regressions is equal
     #with the number of variables that are not in the model
     regAIC = vector('numeric',length(VariablesNotInModel));
    
     #take each variable that is not in the model
     #and include it in the model
     for(i in 1:length(VariablesNotInModel))
     {
        #add the variable to the model
        newRegression = c(bestRegression,VariablesNotInModel[i]);
        #calculate its AIC
        regAIC[i] = getLogisticAIC(response,newRegression,data);
     }
     variableToInclude = VariablesNotInModel[which.min(regAIC)];
     cat('\nforwardSearch :: Step [',stepNumber,'] identified best variable to add ',variableToInclude);
     cat(' which gives AIC = ',regAIC[which.min(regAIC)],'\n');
     
     #the best variable that could be included in the current model
     #is that variable whose inclusion in the current model gives
     #the smallest AIC; however, we include this variable
     #in the model only if it leads to a smaller AIC
     #than the AIC of the current model
     if(min(regAIC)<bestRegressionAIC)
     {
        bestRegression = sort(c(bestRegression,variableToInclude));
        bestRegressionAIC = min(regAIC);
        #delete that variable from the list of variables that are not in the model
        VariablesNotInModel = setdiff(VariablesNotInModel,variableToInclude);
        
        cat('tforwardSearch :: Current model has AIC = ',bestRegressionAIC,' and contains the variables [');
        if(length(bestRegression)>=1)
        {
          for(i in 1:length(bestRegression)) cat(' ',bestRegression[i]);
        }
        cat(']\n');
     }
     else
     {
        #we stop if we did not manage to improve the model by including a variable
        cat('\nforwardSearch :: STOP: could not improve model by adding a variable\n');
        break;
     }
  }
  
  return(list(aic=bestRegressionAIC,reg=bestRegression));
}

#we structure the R code as a C program; this is a choice, not a must
main <- function(datafile)
{
  #read the data
  data = read.table(datafile,header=FALSE);

  #the sample size is 148 (number of rows)
  #the explanatory variables are the first 60 columns
  #the last column is the binary response
  response = ncol(data);
  lastPredictor = ncol(data)-1;

  #perform a forward "greedy" search for the best logistic regression
  #i.e., the logistic regression with the smallest AIC
  forwardResults = forwardSearchAIC(response,data,lastPredictor);
  
  #perform the same forward "greedy" search while avoiding considering
  #a logistic regression whose MLEs are not well determined
  forwardResultsValid = forwardSearchAICvalid(response,data,lastPredictor);
  
  #perform a backward "greedy" search for the best logistic regression
  backwardResults = backwardSearchAIC(response,data,lastPredictor);
  
  #output the results of our searches
  cat('\n\nForward search gives regression with ',length(forwardResults$reg),'explanatory variables [');
  if(length(forwardResults$reg)>=1)
  {
    for(i in 1:length(forwardResults$reg)) cat(' ',forwardResults$reg[i]);
  }
  cat('] with AIC = ',forwardResults$aic,'\n');
  
  cat('\n\nForward search with validity check gives regression with ',length(forwardResultsValid$reg),'explanatory variables [');
  if(length(forwardResultsValid$reg)>=1)
  {
    for(i in 1:length(forwardResultsValid$reg)) cat(' ',forwardResultsValid$reg[i]);
  }
  cat('] with AIC = ',forwardResultsValid$aic,'\n');t
  
  cat('\n\nBackward search gives regression with ',length(backwardResults$reg),'explanatory variables [');
  if(length(backwardResults$reg)>=1)
  {
    for(i in 1:length(backwardResults$reg)) cat(' ',backwardResults$reg[i]);
  }
  cat('] with AIC = ',backwardResults$aic,'\n');
}


#NOTE: YOU NEED THE PACKAGE 'RCDD' TO PROPERLY RUN THIS CODE
#load the 'RCDD' package
library(rcdd);
main('534binarydata.txt');