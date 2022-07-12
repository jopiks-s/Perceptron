#include "Perceptron.h"

float Math::Sigmoid(float x)
{
	return 1 / (1 + pow(E, -x));
}

float Math::DerSigmoid(float x)
{
	return Sigmoid(x) * (1 - Sigmoid(x));
}

float Math::CalculateError(std::vector<float> Out, std::vector<float> Correct)
{
	if (Out.size() != Correct.size()) { std::cout << "Out.size()!=Correct.size()"; return 0; }
	float Error = 0;
	for (int i = 0; i < Out.size(); i++)
		Error += pow(Out[i] - Correct[i], 2);
		
	return Error;
}
