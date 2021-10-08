using SFML.Graphics;
using SFML.System;
using System;


namespace SimpleCNN
{
	//TODO: Надо сделать нормальное обучение
	interface ILayer<InputType, OutputType>
	{
		public OutputType FeedForward(InputType input);

		public InputType BackPropogation(OutputType target);
	}
	//Images
	class ImageScaler : ILayer<Image, Image>//Изначально изображение должно быть больше конечного, иначе будут NaN'ы и все пойдет по одному месту
	{
		Vector2u _scaleTo;

		Image _lastImage;

		public ImageScaler(Vector2u scaleTo)
		{
			_scaleTo = scaleTo;
		}

		public Image BackPropogation(Image target)
		{
			return _lastImage;
		}

		public Image FeedForward(Image image)
		{
			_lastImage = image;
			var ImageSize = image.Size;
			Image ret = new Image(_scaleTo.X, _scaleTo.Y);

			for (uint i = 0; i < _scaleTo.X; i++)
			{
				for (uint ii = 0; ii < _scaleTo.Y; ii++)
				{
					int pixelsCount = 0;

					Vector3f endPixel = new Vector3f();

					for (uint j = (uint)(i * ((float)ImageSize.X / _scaleTo.X)); j < (uint)((i + 1) * ((float)ImageSize.X / _scaleTo.X)); j += 1)
					{
						for (uint jj = (uint)(ii * ((float)ImageSize.Y / _scaleTo.Y)); jj < (uint)((ii + 1) * ((float)ImageSize.Y / _scaleTo.Y)); jj += 1)
						{
							pixelsCount++;

							Color pixel = image.GetPixel(j, jj);

							endPixel += new Vector3f(pixel.R, pixel.G, pixel.B);
						}
					}

					endPixel /= pixelsCount;
					ret.SetPixel(i, ii, new Color((byte)endPixel.X, (byte)endPixel.Y, (byte)endPixel.Z));
				}
			}

			return ret;
		}
	}

	class ImageToTensor : ILayer<Image, Tensor>
	{
		public Image BackPropogation(Tensor target)
		{
			return target.ToImage();
		}

		public Tensor FeedForward(Image input)
		{
			return new Tensor(input);
		}
	}
	//Tensors
	class MaxPoolLayer : ILayer<Tensor, Tensor>
	{
		readonly Vector2i _poolSize;

		readonly int _stepSize;

		Tensor _lastInput;

		Tensor _lastOutput;

		public MaxPoolLayer(Vector2i poolSize, int stepSize)
		{
			_poolSize = poolSize;
			_stepSize = stepSize;
		}

		public Tensor BackPropogation(Tensor target)
		{
			return (Tensor)_lastInput.Clone();//Я вообще не понимаю что тут делать, что именно изменять? Максимальное значение или все из выборки? что так пусть лучше будет так, сильно не повлияет
		}

		public Tensor FeedForward(Tensor input)
		{
			_lastInput = input;
			_lastOutput = input.MaxPool(_poolSize, _stepSize);
			return _lastOutput;
		}
	}

	class AveragePoolLayer : ILayer<Tensor, Tensor>
	{
		readonly Vector2i _poolSize;

		readonly int _stepSize;

		Tensor _lastInput;

		public Tensor BackPropogation(Tensor target)
		{
			return _lastInput;
		}

		public Tensor FeedForward(Tensor input)
		{
			_lastInput = input;
			return input.AveragePool(_poolSize, _stepSize);
		}
	}

	class MaskPoolLayer : ILayer<Tensor, Tensor>
	{
		readonly Map _mask;

		Tensor _lastInput;

		readonly Func<double, double> _activation;

		readonly int _stepSize;

		public MaskPoolLayer(Map mask, Func<double, double> activation, int stepSize)
		{
			_mask = mask;
			_activation = activation;
			_stepSize = stepSize;
		}

		public Tensor BackPropogation(Tensor target)
		{
			return _lastInput;
		}

		public Tensor FeedForward(Tensor input)
		{
			_lastInput = input;
			return input.MaskPool(_mask, _activation, _stepSize);
		}
	}

	class TensorToDoubles : ILayer<Tensor, double[]>
	{
		Tensor _lastInput;

		public Tensor BackPropogation(double[] target)
		{
			return _lastInput;
		}

		public double[] FeedForward(Tensor input)
		{
			_lastInput = (Tensor)input.Clone();
			return input.ToDoubles();
		}
	}
	//NuralNetworks
	class FullConnectedLayer : ILayer<double[], double[]>
	{
		readonly NeuralNetwork _neuralNetwork;

		readonly double _learningRate;

		public FullConnectedLayer(NeuralNetwork neuralNetwork, double learningRate)
		{
			_neuralNetwork = neuralNetwork;
			_learningRate = learningRate;
		}

		public double[] BackPropogation(double[] target)
		{
			return _neuralNetwork.BackPropagation(target, _learningRate);
		}

		public double[] FeedForward(double[] input)
		{
			return _neuralNetwork.FeedForward(input);
		}
	}
	//Rest
	class FillerLayer<FillType> : ILayer<FillType, FillType> //делает целое НИЧЕГО!!!! Нужен для заполнения слоев в сети, если нечего туда вставить
	{
		public FillType BackPropogation(FillType target)
		{
			return target;
		}

		public FillType FeedForward(FillType input)
		{
			return input;
		}
	}

	class UnitedLayer<InputType, HiddenType, OutputType> : ILayer<InputType, OutputType>//Нужно запихать 2 слоя, но место только для одного? Это вам поможет // Вобще всю сеть можно представить как один очень большой двойной слой
	{
		readonly ILayer<InputType, HiddenType> _first;

		readonly ILayer<HiddenType, OutputType> _second;

		public UnitedLayer(ILayer<InputType, HiddenType> first, ILayer<HiddenType, OutputType> second)
		{
			_first = first;
			_second = second;
		}


		public InputType BackPropogation(OutputType target)
		{
			return _first.BackPropogation(_second.BackPropogation(target));
		}

		public OutputType FeedForward(InputType input)
		{
			return _second.FeedForward(_first.FeedForward(input));
		}
	}
}
