using System;

namespace SimpleCNN
{
	interface IFunction
	{
		public double GetOutput(double x);
	}

	class Sigmoid : IFunction
	{
		public double GetOutput(double x)
		{
			return 1 / (1 + Math.Exp(-x));
		}
	}

	class DSigmoid : IFunction
	{
		public double GetOutput(double y)
		{
			return y * (1 - y);
		}
	}
}
