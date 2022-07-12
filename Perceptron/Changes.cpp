#include "Perceptron.h"
using namespace std;

EvolutionChanges::EvolutionChanges() {}

EvolutionChanges::EvolutionChanges(int InputN, int HideLayerN, int HideNeuronN, int OutN)
{
	int LNumber = 2 + HideLayerN;
	using matrix_with_neur = vector<vector<vector<float>>>;
	WChanges = matrix_with_neur(LNumber, vector<vector<float>>());
	WChanges[0] = vector<vector<float>>(1, vector<float>());
	for (int i = 1; i < LNumber; i++)
	{
		if (i == 1)
			WChanges[i] = vector<vector<float>>(HideNeuronN, vector<float>(InputN+1, 0));
		if (i == LNumber - 1)
			WChanges[i] = vector<vector<float>>(OutN, vector<float>(HideNeuronN+1, 0));
		if (i != 1 && i != LNumber - 1)
			WChanges[i] = vector<vector<float>>(HideNeuronN, vector<float>(HideNeuronN+1, 0));
	}
	NeuronDerivatives = vector<vector<float>>(LNumber, vector<float>());
	
}
