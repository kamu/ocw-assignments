function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

%as in regularisation, it is from j >= 1, this skips j=0 (or, in octave j=1)
regvector = ones(1, size(theta)(1));
regvector(1,1) = 0;
%probably should have just used theta(2:end)

regidmat = eye(size(theta)(1));
regidmat(1,1) = 0;

Jreg = (lambda / (2 * m) * (regvector * theta.^2));

J = (1 / m) * ((-y' * log(sigmoid(X * theta))) - ((1 - y)' * log(1 - sigmoid(X * theta)))) + Jreg;

gradreg = (lambda * theta' * regidmat);

grad = (1 / m) * ((sigmoid(X * theta) - y)' * X + gradreg);

% =============================================================

end
