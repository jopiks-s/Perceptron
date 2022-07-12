#include "Perceptron.h"

Layer::Layer(int NeuronN, bool bOutLayer)
{
	for (int i = 0; i < NeuronN; i++)
		NeuronArr.push_back(new Neuron());
	if (bOutLayer)
		return;
	NeuronArr.push_back(new Neuron(true));
}

float Layer::getMaxIndex()
{
	float max=-1, index=-1;
	for (int i = 0; i < NeuronArr.size(); i++)
	{
		if (NeuronArr[i]->Sum > max)
		{
			index = i;
			max = NeuronArr[i]->Sum;
		}
	}
	return index;
}