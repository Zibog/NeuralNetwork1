using System;

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
            public double Charge = 0;

            /// <summary>
            /// Выходной сигнал нейрона
            /// </summary>
            public double Output = 0;

            /// <summary>
            /// Ошибка для данного нейрона
            /// </summary>
            public double Error = 0;

            /// <summary>
            /// Сигнал поляризации
            /// </summary>
            public static double BiasSignal = 1;

            /// <summary>
            /// Вес на сигнале поляризации
            /// </summary>
            public double BiasWeight = 0.01;

            /// <summary>
            /// Количество узлов на предыдущем слое
            /// </summary>
            public int InputLayerSize = 0;

            /// <summary>
            /// Ссылка на предыдущий слой нейронов 
            /// </summary>
            public Node[] InputLayer = null;

            /// <summary>
            /// Вектор входных весов нейрона
            /// </summary>
            public double[] Weights = null;

            /// <summary>
            /// Генератор для инициализации весов
            /// </summary>
            private static Random _randGenerator = new Random();

            /// <summary>
            /// Минимальное значение для начальной инициализации весов
            /// </summary>
            private static double _initMinWeight = -1;

            /// <summary>
            /// Максимальное значение для начальной инициализации весов
            /// </summary>
            private static double _initMaxWeight = 1;

            /// <summary>
            /// Фиктивный нейрон
            /// </summary>
            public Node(Node[] prevLayerNodes)
            {
                InputLayer = prevLayerNodes;

                if (prevLayerNodes == null) return;

                InputLayerSize = prevLayerNodes.Length;
                Weights = new double[InputLayerSize];

                for (int i = 0; i < InputLayerSize; i++)
                {
                    Weights[i] = _initMinWeight + _randGenerator.NextDouble() * (_initMaxWeight - _initMinWeight);
                }
            }

            /// <summary>
            /// Активация нейрона
            /// </summary>
            public void Activate()
            {
                Charge = BiasSignal * BiasWeight;
                for (int i = 0; i < InputLayerSize; i++)
                {
                    Charge += InputLayer[i].Output * Weights[i];
                }

                Output = 1 / (1 + Math.Exp(-Charge));

                Charge = 0;
            }

            /// <summary>
            /// Распространение ошибки на предыдущий слой и пересчёт весов. 
            /// </summary>
            public void ErrorBackPropagation(double learningSpeed)
            {
                Error *= Output * (1 - Output);
                BiasWeight += learningSpeed * Error * BiasSignal;

                for (int i = 0; i < InputLayerSize; i++)
                    InputLayer[i].Error += Error * Weights[i];

                for (int i = 0;i < InputLayerSize; i++)
                    Weights[i] = learningSpeed * Error * InputLayer[i].Output;

                Error = 0;
            }
        }

        public double LearningSpeed = 0.01;
        private Node[] _sensors;
        private Node[][] _layers;
        private Node[] _outputs;

        public StudentNetwork(int[] structure)
        {
            _layers = new Node[structure.Length][];

            _layers[0] = new Node[structure[0]];
            for (int neuron = 0; neuron < structure[0]; neuron++)
                _layers[0][neuron] = new Node(null);

            _sensors = _layers[0];

            for (int layer = 1; layer < structure.Length; layer++)
            {
                _layers[layer] = new Node[structure[layer]];
                for (int neuron = 0; neuron < structure[layer]; neuron++)
                    _layers[layer][neuron] = new Node(_layers[layer - 1]);
            }

            _outputs = _layers[_layers.Length - 1];
        }

        /// <summary>
        /// Прямой проход
        /// </summary>
        private void Forward(Sample sample)
        {
            for (int i = 0; i < sample.input.Length; i++)
                _sensors[i].Output = sample.input[i];

            var lastIndex = _layers.Length - 1;

            for (int i = 1; i < _layers.Length; i++)
                for (int j = 0; j < _layers[i].Length; j++)
                    _layers[i][j].Activate();

            sample.Output = new double[_layers[lastIndex].Length];

            for (int i = 0; i < _layers[lastIndex].Length; i++)
                sample.Output[i] = _layers[lastIndex][i].Output;

            sample.ProcessPrediction(sample.Output);
        }

        /// <summary>
        /// Обратное распространение ошибки
        /// </summary>
        private void BackPropagation(Sample sample, double learningSpeed)
        {
            for (int i = 0; i < _layers[_layers.Length - 1].Length; i++)
                _layers[_layers.Length - 1][i].Error = sample.error[i];

            for (int i = _layers.Length - 1; i >= 0; i--)
                for (int j = 0; j < _layers[i].Length; j++)
                    _layers[i][j].ErrorBackPropagation(learningSpeed);
        }

        public override int Train(Sample sample, double acceptableError, bool parallel)
        {
            var i = 0;
            while (true)
            {
                Forward(sample);

                if (sample.EstimatedError() < acceptableError && sample.Correct())
                    return i;

                BackPropagation(sample, LearningSpeed);

                ++i;
            }
        }

        public override double TrainOnDataSet(SamplesSet samplesSet, int epochsCount, double acceptableError, bool parallel)
        {
            throw new NotImplementedException();
        }

        protected override double[] Compute(double[] input)
        {
            throw new NotImplementedException();
        }
    }
}