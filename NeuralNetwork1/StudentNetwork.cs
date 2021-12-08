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
            public double charge;

            /// <summary>
            /// Выходной сигнал нейрона
            /// </summary>
            public double output;

            /// <summary>
            /// Ошибка для данного нейрона
            /// </summary>
            public double error;

            /// <summary>
            /// Сигнал поляризации (можно и 1 сделать в принципе)
            /// </summary>
            public static double biasSignal = -1.0;

            /// <summary>
            /// Генератор для инициализации весов
            /// </summary>
            private static Random randGenerator = new Random();

            /// <summary>
            /// Минимальное значение для начальной инициализации весов
            /// </summary>
            private static double initMinWeight = -1;//-Math.PI / 2;

            /// <summary>
            /// Максимальное значение для начальной инициализации весов
            /// </summary>
            private static double initMaxWeight = 1;//Math.PI / 2;

            /// <summary>
            /// Количество узлов на предыдущем слое
            /// </summary>
            public int inputLayerSize;

            /// <summary>
            /// Вектор входных весов нейрона
            /// </summary>
            public double[] weights;

            /// <summary>
            /// Вес на сигнале поляризации
            /// </summary>
            public double biasWeight = 0.01;

            /// <summary>
            /// Ссылка на предыдущий слой нейронов 
            /// </summary>
            public Node[] inputLayer;

            /// <summary>
            /// Фиктивный нейрон.
            /// </summary>
            public Node(Node[] prevLayerNodes)
            {
                inputLayer = prevLayerNodes;

                if (prevLayerNodes == null) return;

                inputLayerSize = prevLayerNodes.Length;

                weights = new double[inputLayerSize];

                for (int i = 0; i < weights.Length; ++i)
                {
                    weights[i] = initMinWeight + randGenerator.NextDouble() * (initMaxWeight - initMinWeight);
                }

            }

            /// <summary>
            /// активация нейрона.
            /// </summary>
            public void Activate()
            {

                charge = biasWeight * biasSignal;
                for (int i = 0; i < inputLayer.Length; ++i)
                    charge += inputLayer[i].output * weights[i];

                output = ActivationFunction(charge);

                charge = 0;
            }


            /// <summary>
            /// Распространение ошибки на предыдущий слой и пересчёт весов. 
            /// </summary>
            public void BackpropError(double ita)
            {
                error *= output * (1 - output);

                biasWeight += ita * error * biasSignal;

                for (int i = 0; i < inputLayerSize; i++)
                    inputLayer[i].error += error * weights[i];

                for (int i = 0; i < inputLayerSize; i++)
                    weights[i] += ita * error * inputLayer[i].output;

                error = 0;
            }

            /// <summary>
            /// Функция активации
            /// </summary>
            public static double ActivationFunction(double inp)
            {
                return 1 / (1 + Math.Exp(-inp));//Math.Atan(inp);
            }
        }

        public double LearningSpeed = 0.01;
        private Node[] _sensors;
        private Node[][] _layers;
        private Node[] _outputs;

        public void ReInit(int[] structure, double initialLearningRate = 0.25)
        {
            _layers = new Node[structure.Length][];

            _layers[0] = new Node[structure[0]];
            for (int neuron = 0; neuron < structure[0]; ++neuron)
                _layers[0][neuron] = new Node(null);
            _sensors = _layers[0];

            for (int layer = 1; layer < structure.Length; ++layer)
            {
                _layers[layer] = new Node[structure[layer]];
                for (int neuron = 0; neuron < structure[layer]; ++neuron)
                    _layers[layer][neuron] = new Node(_layers[layer - 1]);
            }
            _outputs = _layers[_layers.Length - 1];
        }

        /// <summary>
        /// Конструктор нейросети – с массивом, определяющим структуру сети
        /// </summary>
        public StudentNetwork(int[] structure)
        {
            ReInit(structure);
        }

        /// <summary>
        /// Прямой проход
        /// </summary>
        public void Run(Sample image)
        {
            for (int i = 0; i < image.input.Length; i++)
                _sensors[i].output = image.input[i];

            for (int i = 1; i < _layers.Length; i++)
                for (int j = 0; j < _layers[i].Length; j++)
                    _layers[i][j].Activate();

            image.Output = new double[_layers[_layers.Length - 1].Length];

            for (int i = 0; i < _layers[_layers.Length - 1].Length; i++)
                image.Output[i] = _layers[_layers.Length - 1][i].output;

            image.recognizedClass = image.ProcessPrediction(image.Output);
        }

        public double[] Run(double[] input)
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
        private void BackProp(Sample image, double ita)
        {
            for (int i = 0; i < _layers[_layers.Length - 1].Length; i++)
                _layers[_layers.Length - 1][i].error = image.error[i];

            var board = _layers.Length - 1;
            Parallel.For(0, board, t =>
            {
                for (int i = _layers.Length - 1; i >= 0; --i)
                    for (int j = 0; j < _layers[i].Length; ++j)
                        _layers[i][j].BackpropError(ita);
            });
        }

        /// <summary>
        /// Обучение одному заданному образу
        /// </summary>
        public override int Train(Sample sample, double acceptableError, bool parallel)
        {
            int i = 0;
            while (i < 100)
            {
                Run(sample);

                //Debug.WriteLine(sample.ToString());
                //Debug.WriteLine("Estimated error : " + sample.EstimatedError());
                
                if (sample.EstimatedError() < acceptableError && sample.Correct())
                {
                    //Debug.WriteLine("Time " + iters);
                    return i;
                }

                ++i;
                BackProp(sample, LearningSpeed);
            }
            return i;
        }

        public override double TrainOnDataSet(SamplesSet samplesSet, int epochsCount, double acceptableError, bool parallel)
        {
            double accuracy = 0;
            double samplesLooked = 0;
            double samplesCount = samplesSet.samples.Count * epochsCount;
            var startTime = DateTime.Now;
            int epochsNum = 1;
            while (epochsNum < epochsCount)
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
                if (accuracy > acceptableError)
                {
                    OnTrainProgress(1, accuracy, DateTime.Now - startTime);
                    return accuracy;
                }
                OnTrainProgress(samplesLooked / samplesCount, accuracy, DateTime.Now - startTime);
                epochsNum++;
            }
            OnTrainProgress(1, accuracy, DateTime.Now - startTime);
            return accuracy;
        }

        protected override double[] Compute(double[] input)
        {
            return Run(input);
        }
    }
}