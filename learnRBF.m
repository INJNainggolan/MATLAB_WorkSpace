%// This is a RBF network trained by BP algorithm   
  
close all; clear; clc;  
  
%%% ************************************************  
%%% ************ step 0: load data ****************  
display('step 0: load data...');  
% train_x = [1 2 3 4 5 6 7 8]; % each sample arranged as a column of train_x  
% train_y = 2 * train_x;  
train_x = rand(5, 10);  
train_y = 2 * train_x;  
test_x = train_x;  
test_y = train_y;  
  
%% from matlab  
% rbf = newrb(train_x, train_y);  
% output = rbf(test_x);  
  
  
%%% ************************************************  
%%% ******** step 1: initialize parameters ********   
display('step 1: initialize parameters...');  
numSamples = size(train_x, 2);  
rbf.inputSize = size(train_x, 1);  
rbf.hiddenSize = numSamples;        % num of Radial Basis function  
rbf.outputSize = size(train_y, 1);  
rbf.alpha = 0.1;  % learning rate (should not be large!)  
  
%% centre of RBF  
for i = 1 : rbf.hiddenSize  
    % randomly pick up some samples to initialize centres of RBF  
    index = randi([1, numSamples]);   
    rbf.center(:, i) =  train_x(:, index);  
end  
  
%% delta of RBF  
rbf.delta = rand(1, rbf.hiddenSize);  
  
%% weight of RBF  
r = 1.0; % random number between [-r, r]  
rbf.weight = rand(rbf.outputSize, rbf.hiddenSize) * 2 * r - r;  
  
  
%%% ************************************************  
%%% ************ step 2: start training ************  
display('step 2: start training...');  
maxIter = 400;  
preCost = 0;  
for i = 1 : maxIter  
    fprintf(1, 'Iteration %d ,', i);  
    rbf = trainRBF(rbf, train_x, train_y);  
    fprintf(1, 'the cost is %d \n', rbf.cost);  
      
    curCost = rbf.cost;  
    if abs(curCost - preCost) < 1e-8  
        disp('Reached iteration termination condition and Termination now!');  
        break;  
    end  
    preCost = curCost;  
end  
  
  
%%% ************************************************  
%%% ************ step 3: start testing ************   
display('step 3: start testing...');  
Green = zeros(rbf.hiddenSize, 1);  
for i = 1 : size(test_x, 2)  
    for j = 1 : rbf.hiddenSize  
        Green(j, 1) = green(test_x(:, i), rbf.center(:, j), rbf.delta(j));  
    end   
    output(:, i) = rbf.weight * Green;  
end  
disp(test_y);  
disp(output);