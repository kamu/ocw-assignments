function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

%theta2 is 10x26

% h\theta(x)
a1 = [ones(m, 1) X]'; % add column of 1s to X

z2 = Theta1 * a1;
a2 = sigmoid(z2);
a2 = [ones(1, size(a2, 2)); a2]; %add row of 1s, because X is transposed

z3 = Theta2 * a2;
a3 = sigmoid(z3)';

%a3 is 5000x10

Y = zeros(size(a3)); % 5000 examples by 10 labels

for i = 1:num_labels
	Y(:, i) = (y == i); % put a 1 in the column for the rows in y that match i. do this for each label
end

J = (1 / m) * sum(sum(-Y .* log(a3) - (1 - Y) .* log(1 - a3)));

% regularisation
J += (lambda / (2 * m)) * (sum(sum(Theta1(:, 2:end) .^ 2)) + sum(sum(Theta2(:, 2:end) .^ 2)));

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

% the feedforward pass has already been performed, so we use those values
delta3 = (a3 - Y)'; %delta3 is 10x5000
sgz2 = sigmoidGradient(z2); %z2 is 25x5000

%Theta2 is 10x26
%remove the bias unit values by 2:end

delta2 = (Theta2(:, 2:end)' * delta3) .* sgz2; % 25x5000

bigdelta1 = delta2 * a1'; %25x401
bigdelta2 = delta3 * a2'; %10x26

reg1 = (lambda / m) * [zeros(size(Theta1, 1),1) Theta1(:,2:end)];
reg2 = (lambda / m) * [zeros(size(Theta2, 1),1) Theta2(:,2:end)];

Theta1_grad = (1 / m) * bigdelta1 + reg1;
Theta2_grad = (1 / m) * bigdelta2 + reg2;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
