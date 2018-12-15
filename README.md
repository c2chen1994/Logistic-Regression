# Logistic-Regression
	Implement Logistic Regression for binary and multiclass classification.
		1
			In this problem, you are given atraining set D = {(xn, yn)N}, where yi ∈ {0, 1} ∀i = 1...N. 
		Important: note that here the binary labels are not −1 or +1, so be very careful about applying formulas.
		The task is to learn the linear model specified by wTx + b that minimizes the logistic loss. Note that we
		do not explicitly append the feature 1 to the data, so you need to explicitly learn the bias/intercept term
		b too. Specifically you need to implement function binary_train in logistic.py which uses gradient
		descent (not stochastic gradient descent) to find the optimal parameters (recall logistic regression does not
		admit a closed-form solution).
			In addition you need to implement function binary_predict in logistic.py. There are two ways of making 
		predictions in logistic regression: deterministic prediction or randomized prediction. Here you need to use 
		the deterministic prediction.
			After finishing implementation, please run logistic_binary.sh which generates logistic_binary.out.

		2
			There are several methods to perform multiclass classification. One of them was one-versus-rest or 
		one-versus-all approach.
			For one-versus-rest classification in a problem with K classes, we need to train K classifiers using a
		black-box. Classifier k is trained on a binary problem, where the two labels corresponds to belonging or
		not belonging to class k. After that, the multiclass prediction is made based on the combination of all
		predictions from K binary classifiers.
			In this problem you will implement one-versus-rest using binary logistic regression (that you have implemented
		in part 1) as the black-box. Important: the way used to predict is to randomized over the classifiers that say “yes”;
		however, here since binary logistic regression naturally predicts a probability for each class (recall the sigmoid model), 
		we will simply predict the class with the highest probability (using numpy argmax).
			To sum up, you need to complete functions OVR_train and OVR_predict to perform one-versus-rest
		classification. After you finished implementation, please run logistic_multiclass.sh script, which
		will produce logistic_multiclass.out.

		3
			Yet another multiclass classification method was multinomial logistic regression. Complete the functions 
		multinomial_train and multinomial_predict to perform multinomial logistic regression, following the same notes 
		as in part 1, that is, 
			1) explicitly learn the biased term; 
			2) perform gradient descent instead of stochastic gradient descent; 
			3) make deterministic predictions.
			After you finished implementation, please run logistic_multiclass.sh script, which will produce
		logistic_multiclass.out.

		Advice 
			We are extensively using softmax and sigmoid function. To avoid numerical issues such as overflow and underflow 
		caused by numpy.exp() and numpy.log(), please use the following implementations:
			• Let x be a input vector to the softmax function. Use x~ = x − max(x) instead of using x directly for the
			  softmax function f. That is, if you want to compute f(x)i, compute f(x~)i = exp(x~i) / sum(exp(x~i)) instead, 
			  which is clearly mathematically equivalent but numerically more stable.
			• If you are using numpy.log(), make sure the input to the log function is positive. Also, there may
			  be chances that one of the outputs of softmax, e.g. f(x~)i, is extremely small but you need the value
			  ln(f(x~)i). In this case you should convert the computation equivalently into x~i − ln(sum(exp(x~i)))
