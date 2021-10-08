using SFML.Graphics;
using SFML.System;
using System;
using System.Collections.Generic;
using System.Linq;

namespace SimpleCNN
{
	class Map : ICloneable
	{
		readonly double[,] _values;

		public double[,] Values => (double[,])_values.Clone();

		public double this[int i, int ii] { get => _values[i, ii]; set => _values[i, ii] = value; }

		public double this[uint i, uint ii] { get => _values[i, ii]; set => _values[i, ii] = value; }

		public readonly int Width;

		public readonly int Height;

		public Map(double[,] values)
		{
			_values = (double[,])values.Clone();
			Width = _values.GetLength(0);
			Height = _values.GetLength(1);
		}

		public Map(int width, int height)
		{
			_values = new double[width, height];
			Width = width;
			Height = height;
		}

		public Map MaxPool(Vector2i poolSize, int stepSize)
		{
			Map map = new Map((Width - poolSize.X + 1) / stepSize, (Height - poolSize.Y + 1) / stepSize);

			for (int i = 0; i < map.Width; i++)
			{
				for (int ii = 0; ii < map.Height; ii++)
				{
					double max = double.MinValue;

					for (int j = 0; j < poolSize.X; j++)
					{
						for (int jj = 0; jj < poolSize.Y; jj++)
						{
							if (max < _values[i * stepSize + j, ii * stepSize + jj])
							{
								max = _values[i * stepSize + j, ii * stepSize + jj];
							}
						}
					}

					map._values[i, ii] = max;
				}
			}

			return map;
		}

		public Map AveragePool(Vector2i poolSize, int stepSize)
		{
			Map map = new Map((Width - poolSize.X + 1) / stepSize, (Height - poolSize.Y + 1) / stepSize);

			for (int i = 0; i < map.Width; i++)
			{
				for (int ii = 0; ii < map.Height; ii++)
				{
					double average = 0;

					for (int j = 0; j < poolSize.X; j++)
					{
						for (int jj = 0; jj < poolSize.Y; jj++)
						{
							average += _values[i * stepSize + j, ii * stepSize + jj];
						}
					}

					average /= poolSize.X * poolSize.Y;

					map._values[i, ii] = average;
				}
			}

			return map;
		}

		public Map SumPool(Vector2i poolSize, int stepSize)
		{
			Map map = new Map((Width - poolSize.X + 1) / stepSize, (Height - poolSize.Y + 1) / stepSize);

			for (int i = 0; i < map.Width; i++)
			{
				for (int ii = 0; ii < map.Height; ii++)
				{
					double sum = 0;

					for (int j = 0; j < poolSize.X; j++)
					{
						for (int jj = 0; jj < poolSize.Y; jj++)
						{
							sum += _values[i * stepSize + j, ii * stepSize + jj];
						}
					}

					map._values[i, ii] = sum;
				}
			}

			return map;
		}

		public Map MaskPool(Map mask, Func<double, double> activation, int stepSize)
		{
			Map map = new Map((Width - mask.Width + 1) / stepSize, (Height - mask.Height + 1) / stepSize);//4x4, 2x2, 2 = 2x2, 6x6 2x2 2 = 3x3, 5x5 3x3 2 = 2x2 // 

			for (int i = 0; i < map.Width; i++)
			{
				for (int ii = 0; ii < map.Height; ii++)
				{
					double sum = 0;

					for (int j = 0; j < mask.Width; j++)
					{
						for (int jj = 0; jj < mask.Height; jj++)
						{
							sum += _values[i * stepSize + j, ii * stepSize + jj] * mask._values[j, jj];
						}
					}

					map._values[i, ii] = activation(sum);
				}
			}

			return map;
		}

		public Map Normalize()
		{
			return MaskPool(new Map(new double[,] { { 1 } }), x => Math.Clamp(x, 0, 1), 1);
		}

		public Map BackPropogation(double[,] target, Map mask, int stepSize, Func<double, double> activation, Func<double, double> deriviate, double learningRate)//возращает новый фильтр
		{
			double[,] output = MaskPool(mask, activation, stepSize)._values;
			double[,] errors = new double[target.GetLength(0), target.GetLength(1)];
			for (int i = 0; i < target.GetLength(0); i++)
			{
				for (int ii = 0; ii < target.GetLength(1); ii++)
				{
					errors[i, ii] = output[i, ii] - target[i, ii];
				}
			}

			double[,] result = (double[,])mask._values.Clone();

			for (int i = 0; i < target.GetLength(0); i++)
			{
				for (int ii = 0; ii < target.GetLength(1); ii++)
				{
					for (int j = 0; j < mask.Width; j++)
					{
						for (int jj = 0; jj < mask.Height; jj++)
						{
							var weightsDelta = errors[i, ii] * deriviate(errors[i, ii]);

							result[j, jj] -= _values[i + j, ii + jj] * weightsDelta * learningRate;
						}
					}
				}
			}

			return new Map(result);
		}

		public object Clone()
		{
			return new Map(_values);
		}

		public static Map operator *(Map left, double right)
		{
			Map map = new Map(left._values);
			for (int i = 0; i < map.Width; i++)
			{
				for (int ii = 0; ii < map.Height; ii++)
				{
					map._values[i, ii] = left._values[i, ii] * right;
				}
			}
			return map;
		}

		public static Map operator /(Map left, double right)
		{
			Map map = new Map(left._values);
			for (int i = 0; i < map.Width; i++)
			{
				for (int ii = 0; ii < map.Height; ii++)
				{
					map._values[i, ii] = left._values[i, ii] / right;
				}
			}
			return map;
		}

		public static Map BorderFilter => new Map(new double[,] { { -1, -1, -1 }, { -1, 8, -1 }, { -1, -1, -1 } });

		public static Map HorizontalBorderFilter => new Map(new double[,] { { -1, -2, -1 }, { 0, 0, 0 }, { 1, 2, 1 } });

		public static Map VerticalBorderFilter => new Map(new double[,] { { -1, 0, 1 }, { -2, 0, 2 }, { -1, 0, 1 } });

		public static Map VerticalFilter => new Map(new double[,] { { 0, 0, 0 }, { 1, 1, 1 }, { 0, 0, 0 } });

		public static Map HorizontalFilter => new Map(new double[,] { { 0, 1, 0 }, { 0, 1, 0 }, { 0, 1, 0 } });
	}

	class Tensor : ICloneable
	{
		readonly Map[] _maps;

		public IEnumerable<Map> Maps => _maps;

		public int Width => _maps[0].Width;

		public int Height => _maps[0].Height;

		public int Depth => _maps.Length;

		public Map this[int i] => _maps[i];

		public Tensor(int width, int height, int depth)
		{
			_maps = new Map[depth];
			for (int i = 0; i < depth; i++)
			{
				_maps[i] = new Map(width, height);
			}
		}

		public Tensor(Image image)
		{
			_maps = new Map[3];

			for (int i = 0; i < 3; i++)
			{
				_maps[i] = new Map((int)image.Size.X, (int)image.Size.Y);
			}

			for (int i = 0; i < image.Size.X; i++)
			{
				for (int ii = 0; ii < image.Size.Y; ii++)
				{
					Color pixel = image.GetPixel((uint)i, (uint)ii);
					_maps[0][i, ii] = pixel.R / 255f;
					_maps[1][i, ii] = pixel.G / 255f;
					_maps[2][i, ii] = pixel.B / 255f;
				}
			}
		}

		public Tensor(Map[] maps)
		{
			_maps = new Map[maps.Length];
			for (int i = 0; i < _maps.Length; i++)
			{
				_maps[i] = (Map)maps[i].Clone();
			}
		}

		public Tensor Scale(Vector2i scaleTo)
		{
			Tensor result = new Tensor(scaleTo.X, scaleTo.Y, _maps.Length);

			for (int map = 0; map < _maps.Length; map++)
			{
				for (int i = 0; i < scaleTo.X; i++)
				{
					for (int ii = 0; ii < scaleTo.Y; ii++)
					{
						int pixelsCount = 0;

						double end = 0;

						for (uint j = (uint)(i * ((float)Width / scaleTo.X)); j < (uint)((i + 1) * ((float)Width / scaleTo.X)); j += 1)
						{
							for (uint jj = (uint)(ii * ((float)Height / scaleTo.Y)); jj < (uint)((ii + 1) * ((float)Height / scaleTo.Y)); jj += 1)
							{
								pixelsCount++;

								end += _maps[map][j, jj];
							}
						}

						result._maps[map][i, ii] = end / pixelsCount;
					}
				}
			}

			return result;
		}

		public Tensor MaxPool(Vector2i poolSize, int stepSize)
		{
			Map[] maps = new Map[_maps.Length];

			for (int i = 0; i < _maps.Length; i++)
			{
				maps[i] = _maps[i].MaxPool(poolSize, stepSize);
			}

			return new Tensor(maps);
		}

		public Tensor AveragePool(Vector2i poolSize, int stepSize)
		{
			Map[] maps = new Map[_maps.Length];

			for (int i = 0; i < _maps.Length; i++)
			{
				maps[i] = _maps[i].AveragePool(poolSize, stepSize);
			}

			return new Tensor(maps);
		}

		public Tensor SumPool(Vector2i poolSize, int stepSize)
		{
			Map[] maps = new Map[_maps.Length];

			for (int i = 0; i < _maps.Length; i++)
			{
				maps[i] = _maps[i].SumPool(poolSize, stepSize);
			}

			return new Tensor(maps);
		}

		public Tensor MaskPool(Map mask, Func<double, double> activation, int stepSize)
		{
			Map[] maps = new Map[_maps.Length];

			for (int i = 0; i < _maps.Length; i++)
			{
				maps[i] = _maps[i].MaskPool(mask, activation, stepSize);
			}

			return new Tensor(maps);
		}

		public Tensor MaskPool(Tensor masks, Func<double, double> activation, int stepSize)
		{
			List<Map> result = new List<Map>();
			for (int i = 0; i < masks._maps.Length; i++)
			{
				result.AddRange(MaskPool(masks._maps[i], activation, stepSize)._maps);
			}
			return new Tensor(result.ToArray());
		}

		public Tensor Normalize()
		{
			return MaskPool(new Map(new double[,] { { 1 } }), x => Math.Clamp(x, 0, 1), 1);
		}

		public Tensor Concat(Tensor tensor)
		{
			List<Map> result = _maps.ToList();
			result.AddRange(tensor._maps);
			return new Tensor(result.ToArray());
		}

		public double[] ToDoubles()
		{
			List<double> input = new List<double>();
			for (int i = 0; i < _maps.Length; i++)
			{
				for (int ii = 0; ii < _maps[i].Width; ii++)
				{
					for (int iii = 0; iii < _maps[i].Width; iii++)
					{
						input.Add(_maps[i][ii, iii]);
					}
				}
			}
			return input.ToArray();
		}

		public Image ToImage()
		{
			Image result = new Image((uint)_maps[0].Width, (uint)_maps[0].Height);
			for (uint i = 0; i < result.Size.X; i++)
			{
				for (uint ii = 0; ii < result.Size.Y; ii++)
				{
					result.SetPixel(i, ii, new Color((byte)Math.Clamp(_maps[0][i, ii] * 255, 0, 255), (byte)Math.Clamp(_maps[1][i, ii] * 255, 0, 255), (byte)Math.Clamp(_maps[2][i, ii] * 255, 0, 255)));
				}
			}
			return result;
		}

		public object Clone()
		{
			return new Tensor(_maps);
		}
	}
}
