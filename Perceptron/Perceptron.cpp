#include "Perceptron.h"
#include <cmath>

void Perceptron::DefaultConstruct()
{
	LayersArr.push_back(new Layer(InputN));

	for (int i = 0; i < HideLayerN; i++)
		LayersArr.push_back(new Layer(HideNeuronN));

	LayersArr.push_back(new Layer(OutN, true));

	FillRandomWeigths();
}

Perceptron::Perceptron(int InputN, int HideLayerN, int HideNeuronN, int OutN, float LearningRate)
	:InputN(InputN), HideLayerN(HideLayerN), HideNeuronN(HideNeuronN), OutN(OutN), LearningRate(LearningRate)
{
	DefaultConstruct();
}

Perceptron::Perceptron(std::string Path)
{
	std::ifstream fin(Path);
	if (!fin.is_open())
	{
		std::cout << "ERROR:Incorrect path for ai creation!";
		return;
	}
	std::stringstream strmPath(Path);
	std::string buff = Path;
	while (std::getline(strmPath, buff, '\\')) {}

	std::stringstream strmPath1(buff);
	std::getline(strmPath1, buff, '.');
	std::getline(strmPath1, buff, '.');
	if (buff != "ai")
	{
		std::cout << "ERROR:Incorrect file type!";
		return;
	}

	std::getline(fin, buff, '\n');
	InputN = std::stoi(buff);
	std::getline(fin, buff, '\n');
	HideLayerN = std::stoi(buff);
	std::getline(fin, buff, '\n');
	HideNeuronN = std::stoi(buff);
	std::getline(fin, buff, '\n');
	OutN = std::stoi(buff);
	std::getline(fin, buff, '\n');
	LearningRate = std::stof(buff);
	DefaultConstruct();

	//////////////// SET SAVED Weights

	for (int i = 1; i < LayersArr.size(); i++)
	{
		int prevLayerN = LayersArr[i - 1]->NeuronArr.size();
		for (int NeurInd = 0; NeurInd < LayersArr[i]->NeuronArr.size(); NeurInd++)
		{
			std::getline(fin, buff, '\n');
			if (LayersArr[i]->NeuronArr[NeurInd]->bShiftNeur)
			{
				std::getline(fin, buff, '\n');
				continue;
			}
			for (int w = 0; w < prevLayerN; w++)
			{
				std::getline(fin, buff, '\n');
				LayersArr[i]->NeuronArr[NeurInd]->Weights[w] = std::stof(buff);
			}
		}
	}
}

void Perceptron::FillRandomWeigths()
{
	srand(time(NULL));

	if (LayersArr.size() < 2)
		return;

	for (int i = 1; i < LayersArr.size(); i++)
	{
		int PrevN = LayersArr[i - 1]->NeuronArr.size();
		std::vector<Neuron*>* CurrNeuronArr = &(LayersArr[i]->NeuronArr);
		for (int k = 0; k < CurrNeuronArr->size(); k++)
		{
			if (CurrNeuronArr->at(k)->bShiftNeur)
				continue;
			for (int j = 0; j < PrevN; j++)
			{
				float RandW = (rand() % 4001 - 2000) / 1000.0;
				CurrNeuronArr->at(k)->Weights.push_back(RandW);
			}
		}
	}
}

void Perceptron::Teach(std::vector<std::vector<float>> InputData, std::vector<std::vector<float>> CorrectData, std::string Inf)
{
	int TeachTimes = InputData.size();
	std::vector<EvolutionChanges> AllChanges;
	for (int i = 0; i < TeachTimes; i++)
	{
		AllChanges.push_back(Forw_Backw_Propagation(InputData[i], CorrectData[i]));
	}
	EvolutionChanges AverageChange(InputN, HideLayerN, HideNeuronN, OutN);

	for (int i = 0; i < AllChanges.size(); i++)
		for (int k = 1; k < AllChanges[i].WChanges.size(); k++)
			for (int t = 0; t < AllChanges[i].WChanges[k].size(); t++)
				for (int w = 0; w < AllChanges[i].WChanges[k][t].size(); w++)
					AverageChange.WChanges[k][t][w] += AllChanges[i].WChanges[k][t][w];


	for (int i = 0; i < AverageChange.WChanges.size(); i++)
		for (int k = 0; k < AverageChange.WChanges[i].size(); k++)
			for (int w = 0; w < AverageChange.WChanges[i][k].size(); w++)
				AverageChange.WChanges[i][k][w] /= TeachTimes;

	for (int i = 1; i < LayersArr.size(); i++)
	{
		int prevLayerN = LayersArr[i - 1]->NeuronArr.size();
		for (int NeurInd = 0; NeurInd < LayersArr[i]->NeuronArr.size(); NeurInd++)
		{
			if (LayersArr[i]->NeuronArr[NeurInd]->bShiftNeur)
				continue;
			for (int w = 0; w < prevLayerN; w++)
				LayersArr[i]->NeuronArr[NeurInd]->Weights[w] -= AverageChange.WChanges[i][NeurInd][w];
		}
	}
}

EvolutionChanges Perceptron::Forw_Backw_Propagation(std::vector<float> Input, std::vector<float> CorrectData)
{
	if (LayersArr.size() < 2)
	{
		std::cout << "ERROR at ForwardPropagation: LayersArr.size() < 2"; return EvolutionChanges();
	}
	if (CorrectData.size() != OutN)
	{
		std::cout << "ERROR at ForwardPropagation: CorrectData.size() != OutN"; return EvolutionChanges();
	}
	if (Input.size() != LayersArr[0]->NeuronArr.size() - 1)
	{
		std::cout << "ERROR at ForwardPropagation: Input size() != LayersArr InputN"; return EvolutionChanges();
	}

	//FORWARD
	std::vector<float> Res = Forw_Propagation(Input);


	//float Err = Math::CalculateError(Res, CorrectData);
	////std::cout << "RESAULT\n";
	////for (int i=0;i<Res.size();i++)
	////	std::cout << "["<<i<<"]: "<<Res[i] << '\n';
	//std::cout << "TOTAL ERROR: " << Err << '\n';
	//if (Err > 1)
	//	for (int a = 0; a < 10; a++)
	//		if (CorrectData[a])
	//			std::cout << a;
	//std::cout << '\n';


	//BACKWARD
	EvolutionChanges Ev(InputN, HideLayerN, HideNeuronN, OutN);
	for (int i = LayersArr.size() - 1; i >= 1; i--)
	{
		for (int NeurInd = 0; NeurInd < LayersArr[i]->NeuronArr.size(); NeurInd++)
		{
			Neuron* CurrNeuron = LayersArr[i]->NeuronArr[NeurInd];
			if (CurrNeuron->bShiftNeur)
				continue;

			if (i == LayersArr.size() - 1)
			{
				Ev.NeuronDerivatives[i].push_back(2 * (CurrNeuron->Sum - CorrectData[NeurInd]));
			}
			else
			{
				float NeurDer = 0;
				for (int ni = 0; ni < LayersArr[i + 1]->NeuronArr.size(); ni++)
				{
					Neuron* NextNeur = LayersArr[i + 1]->NeuronArr[ni];
					if (NextNeur->bShiftNeur)
						continue;
					NeurDer += NextNeur->Weights[NeurInd] * Math::DerSigmoid(NextNeur->InitSum) * Ev.NeuronDerivatives[i + 1][ni];
				}
				Ev.NeuronDerivatives[i].push_back(NeurDer);
			}

			for (int w = 0; w < CurrNeuron->Weights.size(); w++)
			{
				float WChange =
					LearningRate * LayersArr[i - 1]->NeuronArr[w]->Sum * Math::DerSigmoid(CurrNeuron->InitSum) * Ev.NeuronDerivatives[i][NeurInd];
				Ev.WChanges[i][NeurInd][w] = WChange;
			}
		}
	}

	return Ev;
}

std::vector<float> Perceptron::Forw_Propagation(std::vector<float> Input)
{
	if (LayersArr.size() < 2)
	{
		std::cout << "ERROR at Forw_Propagation: LayersArr.size() < 2"; return std::vector<float>();
	}

	if (Input.size() != LayersArr[0]->NeuronArr.size() - 1)
	{
		std::cout << "ERROR at Forw_Propagation: Input size() != LayersArr InputN"; return std::vector<float>();
	}

	for (int i = 0; i < LayersArr[0]->NeuronArr.size(); i++)
	{
		if (LayersArr[0]->NeuronArr[i]->bShiftNeur)
			continue;
		LayersArr[0]->NeuronArr[i]->Sum = Input[i];
	}

	for (int i = 1; i < LayersArr.size(); i++)
	{
		for (Neuron* CurrNeuron : LayersArr[i]->NeuronArr)
		{
			if (CurrNeuron->bShiftNeur)
				continue;
			CurrNeuron->Sum = 0;
			CurrNeuron->InitSum = 0;
			for (int WeightInd = 0; WeightInd < CurrNeuron->Weights.size(); WeightInd++)
			{
				float prevSum = LayersArr[i - 1]->NeuronArr[WeightInd]->Sum;
				CurrNeuron->InitSum += prevSum * CurrNeuron->Weights[WeightInd];
			}
			CurrNeuron->Sum = Math::Sigmoid(CurrNeuron->InitSum);
		}
	}
	return GetColumnValues(LayersArr.size() - 1);
}

void Perceptron::SaveNetwork(std::string Path)
{
	bool findSame = false;
	int i = 0;
	do {
		findSame = false;
		std::string Copy = Path;
		Copy += "SaveAI_" + std::to_string(i) + ".ai";
		if (std::ifstream(Copy).is_open())
		{
			i++;
			findSame = true;
		}

	} while (findSame);
	Path += "SaveAI_" + std::to_string(i) + ".ai";
	std::ofstream fon(Path);
	fon << InputN << '\n';
	fon << HideLayerN << '\n';
	fon << HideNeuronN << '\n';
	fon << OutN << '\n';
	fon << LearningRate << '\n';
	for (int i = 1; i < LayersArr.size(); i++)
	{
		int prevLayerN = LayersArr[i - 1]->NeuronArr.size();
		for (int NeurInd = 0; NeurInd < LayersArr[i]->NeuronArr.size(); NeurInd++)
		{
			fon << i << " " << NeurInd << '\n';
			if (LayersArr[i]->NeuronArr[NeurInd]->bShiftNeur)
			{
				fon << "Shift Neuron" << '\n';
				continue;
			}
			std::vector<float> CopyW = LayersArr[i]->NeuronArr[NeurInd]->Weights;
			for (int w = 0; w < prevLayerN; w++)
				fon << CopyW[w] << '\n';
		}
	}

}

std::vector<float> Perceptron::GetColumnValues(int IndexL)
{
	std::vector<float> ret;
	if ((!(IndexL >= 0 && IndexL < LayersArr.size())) || LayersArr.size() < 2) { std::cout << "ERROR:Perceptron::GetColumnValues [0]"; return ret; }
	for (int i = 0; i < LayersArr[IndexL]->NeuronArr.size(); i++)
		ret.push_back(LayersArr[IndexL]->NeuronArr[i]->Sum);
	return ret;
}