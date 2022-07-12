#include "Perceptron.h"

Neuron::Neuron(bool ShiftNeur):Sum(0), InitSum(0), bShiftNeur(ShiftNeur)
{	
	if (ShiftNeur) { Sum = 1; InitSum = 1; }
}

