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
            }
        }
        public StudentNetwork(int[] structure)
        {
            // TODO
        }

        public override int Train(Sample sample, double acceptableError, bool parallel)
        {
            throw new NotImplementedException();
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