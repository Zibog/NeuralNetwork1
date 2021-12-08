using System;
using System.Linq;
using System.Threading.Tasks;

namespace NeuralNetwork1
{
    public class StudentNetwork : BaseNetwork
    {
        /// <summary>
        /// Один нейрон сети
        /// </summary>
        private class Node
        {
            /// <summary>
            /// Входной взвешенный сигнал нейрона
            /// </summary>
            private double _charge;

            /// <summary>
            /// Выходной сигнал нейрона
            /// </summary>
            public double output;

            /// <summary>
            /// Ошибка для данного нейрона
            /// </summary>
            public double error;

            /// <summary>
            /// Сигнал поляризации
            /// </summary>
            private const double BiasSignal = -1.0;

            /// <summary>
            /// Генератор для инициализации весов
            /// </summary>
            private static Random _r = new Random();

            /// <summary>
            /// Минимальное значение для начальной инициализации весов
            /// </summary>
            private static double _initMinWeight = -1;//-Math.PI / 2;

            /// <summary>
            /// Максимальное значение для начальной инициализации весов
            /// </summary>
            private static double _initMaxWeight = 1;//Math.PI / 2;

            /// <summary>
            /// Количество узлов на предыдущем слое
            /// </summary>
            private int _inputLayerSize;

            /// <summary>
            /// Вектор входных весов нейрона
            /// </summary>
            private double[] _weights;

            /// <summary>
            /// Вес на сигнале поляризации
            /// </summary>
            private double _biasWeight = 0.01;

            /// <summary>
            /// Ссылка на предыдущий слой нейронов 
            /// </summary>
            private Node[] _prevLayer;

            /// <summary>
            /// Фиктивный нейрон
            /// </summary>
            public Node(Node[] prevLayerNodes)
            {
                _prevLayer = prevLayerNodes;

                if (prevLayerNodes == null) return;

                _inputLayerSize = prevLayerNodes.Length;
                _weights = new double[_inputLayerSize];
                for (int i = 0; i < _weights.Length; ++i)
                    _weights[i] = _initMinWeight + _r.NextDouble() * (_initMaxWeight - _initMinWeight);
            }

            /// <summary>
            /// Активация нейрона
            /// </summary>
            public void Activate()
            {
                _charge = _biasWeight * BiasSignal;
                for (int i = 0; i < _prevLayer.Length; ++i)
                    _charge += _prevLayer[i].output * _weights[i];
                output = ActivationFunction(_charge);
                
                _charge = 0;
            }


            /// <summary>
            /// Распространение ошибки на предыдущий слой и пересчёт весов
            /// </summary>
            public void BackPropagation(double learningRate)
            {
                error *= output * (1 - output);
                _biasWeight += learningRate * error * BiasSignal;

                for (int i = 0; i < _inputLayerSize; i++)
                    _prevLayer[i].error += error * _weights[i];

                for (int i = 0; i < _inputLayerSize; i++)
                    _weights[i] += learningRate * error * _prevLayer[i].output;

                error = 0;
            }

            /// <summary>
            /// Функция активации
            /// </summary>
            private static double ActivationFunction(double value) => 1 / (1 + Math.Exp(-value));//Math.Atan(value);
        }

        private const double LearningRate = 0.01;
        private Node[] _sensors;
        private Node[][] _layers;
        private Node[] _outputs;

        /// <summary>
        /// Конструктор нейросети
        /// </summary>
        public StudentNetwork(int[] structure)
        {
            _layers = new Node[structure.Length][];

            // Обработка входного слоя
            _layers[0] = new Node[structure[0]];
            for (int neuron = 0; neuron < structure[0]; ++neuron)
                _layers[0][neuron] = new Node(null);
            _sensors = _layers[0];

            // Обработка остальных слоёв
            for (int layer = 1; layer < structure.Length; ++layer)
            {
                _layers[layer] = new Node[structure[layer]];
                for (int neuron = 0; neuron < structure[layer]; ++neuron)
                    _layers[layer][neuron] = new Node(_layers[layer - 1]);
            }
            _outputs = _layers[_layers.Length - 1];
        }

        /// <summary>
        /// Прямой проход
        /// </summary>
        private void Run(Sample sample)
        {
            for (int i = 0; i < sample.input.Length; i++)
                _sensors[i].output = sample.input[i];

            for (int i = 1; i < _layers.Length; i++)
                for (int j = 0; j < _layers[i].Length; j++)
                    _layers[i][j].Activate();

            sample.Output = new double[_layers[_layers.Length - 1].Length];

            for (int i = 0; i < sample.Output.Length; i++)
                sample.Output[i] = _layers[_layers.Length - 1][i].output;

            sample.ProcessPrediction(sample.Output);
        }

        /// <summary>
        /// Прямой проход
        /// </summary>
        private double[] Run(double[] input)
        {
            for (int i = 0; i < input.Length; i++)
                _sensors[i].output = input[i];

            for (int i = 1; i < _layers.Length; i++)
                for (int j = 0; j < _layers[i].Length; j++)
                    _layers[i][j].Activate();

            var output = new double[_layers[_layers.Length - 1].Length];

            for (int i = 0; i < _layers[_layers.Length - 1].Length; i++)
                output[i] = _layers[_layers.Length - 1][i].output;

            return output;
        }

        /// <summary>
        /// Обратное распространение ошибки
        /// </summary>
        private void BackPropagation(Sample sample)
        {
            for (int i = 0; i < _layers[_layers.Length - 1].Length; i++)
                _layers[_layers.Length - 1][i].error = sample.error[i];

            var board = _layers.Length - 1;
            Parallel.For(0, board, t =>
            {
                for (int i = _layers.Length - 1; i >= 0; --i)
                    for (int j = 0; j < _layers[i].Length; ++j)
                        _layers[i][j].BackPropagation(LearningRate);
            });
        }

        public override int Train(Sample sample, double acceptableError, bool parallel)
        {
            int i = 0;
            while (i < 100)
            {
                Run(sample);
                
                if (sample.EstimatedError() < acceptableError && sample.Correct())
                    return i;

                BackPropagation(sample);
                ++i;
            }
            return i;
        }

        public override double TrainOnDataSet(SamplesSet samplesSet, int epochsCount, double acceptableError, bool parallel)
        {
            double accuracy = 0;
            int samplesLooked = 0;
            double samplesCount = samplesSet.samples.Count * epochsCount;
            var startTime = DateTime.Now;
            while (epochsCount > 0)
            {
                double samplesCorrect = 0;
                for (int i = 0; i < samplesSet.samples.Count; i++)
                {
                    if (Train(samplesSet.samples.ElementAt(i), acceptableError, parallel) == 0)
                        ++samplesCorrect;
                    ++samplesLooked;
                    if (samplesLooked % 25 == 0)
                        OnTrainProgress(samplesLooked / samplesCount, accuracy, DateTime.Now - startTime);
                }
                accuracy = samplesCorrect / samplesSet.samples.Count;
                if (accuracy >= 1 - acceptableError - 1e-10)
                {
                    OnTrainProgress(1, accuracy, DateTime.Now - startTime);
                    return accuracy;
                }
                OnTrainProgress(samplesLooked / samplesCount, accuracy, DateTime.Now - startTime);
                --epochsCount;
            }
            OnTrainProgress(1, accuracy, DateTime.Now - startTime);
            return accuracy;
        }

        protected override double[] Compute(double[] input) => Run(input);
    }
}