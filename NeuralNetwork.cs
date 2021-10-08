using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace SimpleCNN
{
	static class Statics
	{
		public static Random Random = new Random();
	}
	//взял всю это с SimpleNN (Onigiri) и переписал на C#, потому что было лень самому делать
	class NeuralLayer
	{
		public readonly int Size;
		public readonly double[] Neurons;
		public readonly double[] Biases;
		public readonly double[,] Weights;

		public NeuralLayer(int size, int nextSize)
		{
			Size = size;
			Neurons = new double[size];
			Biases = new double[size];
			Weights = new double[size, nextSize];
		}

		public NeuralLayer(byte[] bytes)
		{
			Weights = new double[BitConverter.ToInt64(bytes.Take(8).ToArray()), BitConverter.ToInt64(bytes.Skip(8).Take(8).ToArray())];

			var selfSize = Weights.GetLength(0);
			var nextSize = Weights.GetLength(1);

			for (int i = 0; i < selfSize; i++)
			{
				for (int ii = 0; ii < nextSize; ii++)
				{
					Weights[i, ii] = BitConverter.ToDouble(bytes.Skip(8 * (2 + i * nextSize + ii)).Take(8).ToArray());
				}
			}

			Biases = new double[selfSize];

			for (int i = 0; i < selfSize; i++)
			{
				Biases[i] = BitConverter.ToDouble(bytes.Skip(8 * (2 + selfSize * nextSize + i)).Take(8).ToArray());
			}

			Size = selfSize;
			Neurons = new double[Size];
		}

		public byte[] GetBytes()
		{
			List<byte> result = new List<byte>(8 * (2 + (Weights.GetLength(0) * Weights.GetLength(1)) + Biases.Length));//(разметочный, весы, биасы) * 8 (байтов на 1 число)
			result.AddRange(BitConverter.GetBytes((long)Weights.GetLength(0)));
			result.AddRange(BitConverter.GetBytes((long)Weights.GetLength(1)));

			for (int i = 0; i < Weights.GetLength(0); i++)
			{
				for (int ii = 0; ii < Weights.GetLength(1); ii++)
				{
					result.AddRange(BitConverter.GetBytes(Weights[i, ii]));
				}
			}

			for (int i = 0; i < Weights.GetLength(0); i++)
			{
				result.AddRange(BitConverter.GetBytes(Biases[i]));
			}

			return result.ToArray();
		}
	}

	class NeuralNetwork
	{
		private readonly NeuralLayer[] _layers;
		public int LayerCount => _layers.Length;
		private readonly IFunction _activation;
		private readonly IFunction _derivative;

		public NeuralNetwork(IFunction activation, IFunction derivative, params int[] sizes)
		{
			_activation = activation;
			_derivative = derivative;
			_layers = new NeuralLayer[sizes.Length];
			for (int i = 0; i < sizes.Length; i++)
			{
				int nextSize = 0;
				if (i < sizes.Length - 1) nextSize = sizes[i + 1];
				_layers[i] = new NeuralLayer(sizes[i], nextSize);
				for (int j = 0; j < sizes[i]; j++)
				{
					if (i != 0)
					{
						_layers[i].Biases[j] = Statics.Random.NextDouble() * 2.0 - 1.0;
					}
					for (int k = 0; k < nextSize; k++)
					{
						_layers[i].Weights[j, k] = Statics.Random.NextDouble() * 2.0 - 1.0;
					}
				}
			}
		}

		public NeuralNetwork(string path)
		{
			List<NeuralLayer> buff = new List<NeuralLayer>();

			_activation = new Sigmoid();
			_derivative = new DSigmoid();

			for (int fileIndex = 0; File.Exists(path + $"layer_{fileIndex}.save"); fileIndex++)
			{
				FileStream stream = File.Open(path + $"layer_{fileIndex}.save", FileMode.Open);

				using (stream)
				{
					List<byte> fileInBytes = new List<byte>();

					byte[] buffer = new byte[8];
					int readedCount = 0;
					while (readedCount < stream.Length)
					{
						buffer = new byte[Math.Min(4096, stream.Length - readedCount)];

						stream.Read(buffer, 0, buffer.Length);

						fileInBytes.AddRange(buffer);
						readedCount += buffer.Length;
					}

					buff.Add(new NeuralLayer(fileInBytes.ToArray()));
				}
			}

			_layers = buff.ToArray();
		}

		public double[] FeedForward(double[] inputs)
		{
			inputs.CopyTo(_layers[0].Neurons, 0);
			for (int i = 1; i < _layers.Length; i++)
			{
				NeuralLayer currentLayer = _layers[i - 1];
				NeuralLayer nextLayer = _layers[i];
				for (int j = 0; j < nextLayer.Size; j++)
				{
					nextLayer.Neurons[j] = 0;
					for (int k = 0; k < currentLayer.Size; k++)
					{
						nextLayer.Neurons[j] += currentLayer.Neurons[k] * currentLayer.Weights[k, j];
					}
					nextLayer.Neurons[j] += nextLayer.Biases[j];
					nextLayer.Neurons[j] = _activation.GetOutput(nextLayer.Neurons[j]);
				}
			}
			return _layers[^1].Neurons;
		}

		public double[] BackPropagation(double[] targets, double learningRate)
		{
			double[] errors = new double[_layers[^1].Size];
			for (int i = 0; i < _layers[^1].Size; i++)
			{
				errors[i] = targets[i] - _layers[^1].Neurons[i];
			}
			for (int k = _layers.Length - 2; k >= 0; k--)
			{
				NeuralLayer previuosLayer = _layers[k];
				NeuralLayer currentLayer = _layers[k + 1];
				double[] errorsNext = new double[previuosLayer.Size];
				double[] gradients = new double[currentLayer.Size];
				for (int i = 0; i < currentLayer.Size; i++)
				{
					gradients[i] = errors[i] * _derivative.GetOutput(_layers[k + 1].Neurons[i]);
					gradients[i] *= learningRate;
				}
				double[,] deltas = new double[currentLayer.Size, previuosLayer.Size];
				for (int i = 0; i < currentLayer.Size; i++)
				{
					for (int j = 0; j < previuosLayer.Size; j++)
					{
						deltas[i, j] = gradients[i] * previuosLayer.Neurons[j];
					}
				}
				for (int i = 0; i < previuosLayer.Size; i++)
				{
					errorsNext[i] = 0;
					for (int j = 0; j < currentLayer.Size; j++)
					{
						errorsNext[i] += previuosLayer.Weights[i, j] * errors[j];
					}
				}
				errors = new double[previuosLayer.Size];
				errorsNext.CopyTo(errors, 0);
				for (int i = 0; i < currentLayer.Size; i++)
				{
					for (int j = 0; j < previuosLayer.Size; j++)
					{
						previuosLayer.Weights[j, i] = previuosLayer.Weights[j, i] + deltas[i, j];
					}
				}
				for (int i = 0; i < currentLayer.Size; i++)
				{
					currentLayer.Biases[i] += gradients[i];
				}
			}
			return errors;
		}

		public void SaveToFiles(string path)
		{
			for (int i = 0; i < _layers.Length; i++)
			{
				using var stream = File.Open(path + $"layer_{i}.save", FileMode.Create);
				{
					stream.Write(_layers[i].GetBytes());

					stream.Flush();
				}
			}
		}
	}

	//TODO: Сохранение сверточной нейронной сети
	class ConvolutionNeuralNetwork<InputType, OutputType> : ILayer<InputType, OutputType>
	{
		readonly ILayer<InputType, OutputType> _represent;/* Можно было бы сделать что-то умное, но мне так насрать. 
		                                          * Что так будет 1 слой, который будет представлять всю сеть (объединенный слой вам в помощь).
		                                          * И вообще всю сеть я тоже привел к слою, потому что могу
		                                          */

		public ConvolutionNeuralNetwork(ILayer<InputType, OutputType> represent)
		{
			_represent = represent;
		}

		public InputType BackPropogation(OutputType target)
		{
			return _represent.BackPropogation(target);
		}

		public OutputType FeedForward(InputType input)
		{
			return _represent.FeedForward(input);
		}
	}
}
