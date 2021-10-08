using SFML.Graphics;
using SFML.System;

namespace SimpleCNN
{
	class Program
	{
		static void Main()
		{
			// Создание нейронной сети c 2 входами и 1 выходом
			// Первый параметр - функция активации
			// Второй параметр - её производная
			// Дальше размеры слоев сети
			var NN = new NeuralNetwork(new Sigmoid(), new DSigmoid(), 2, 1);

			// Сохранение сети в файлы, по одному на каждый слой
			NN.SaveToFiles("save/");

			// Загрузка сети из файлов, NN1 == NN
			var NN1 = new NeuralNetwork("save/");

			// Проход по сети с результатом
			double[] output = NN.FeedForward(new double[] { 1, 1 });

			// Проход по сети в обратном порядке с обучением весов
			// Первый параметр - выход который мы хотим получить
			// Второй параметр - скорость обучения
			// Возращает ошибку входа
			double[] nextTarget = NN.BackPropagation(new double[] { 1 }, 0.01);

			// Создание сверточной нейронной сети
			var CNN = new ConvolutionNeuralNetwork<Image, double[]>(
				new UnitedLayer<Image, Tensor, double[]>(
					new UnitedLayer<Image, Image, Tensor>(

						new ImageScaler(new Vector2u(10, 10)), 
						new ImageToTensor()),	new UnitedLayer<Tensor, double[], double[]>(
						new TensorToDoubles(), 
						new FullConnectedLayer(new NeuralNetwork(new Sigmoid(), new DSigmoid(), 100, 2), 0.01)
						)
					)
				);
			// Выглядит громоздко, на если присмотрется то все довольно легко
			// Все UnitedLayer'ы можно сразу откинуть
			// Смотрим на самые нижние слои, а именно ImageScaler, ImageToTensor, TensorToDoubles, FullConnectedLayer
			// По названиям и возрощаемым типам можно легко понять, что это делает
			// Сначало мы уменьшаем изображение до нужного нам, потому его преобразуем в тензор, тензор преобразуем в массив чисел и его подаем на полностью соединеный слой(обычную нейронную сеть)
		}
	}
}
