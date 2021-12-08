using System;
using System.Collections.Generic;
using System.Diagnostics;
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
            public double charge = 0;

            /// <summary>
            /// Выходной сигнал нейрона
            /// </summary>
            public double output = 0;

            /// <summary>
            /// Ошибка для данного нейрона
            /// </summary>
            public double error = 0;

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
            private static double initMinWeight = -1;//-Math.PI / 2 //-1;

            /// <summary>
            /// Максимальное значение для начальной инициализации весов
            /// </summary>
            private static double initMaxWeight = 1;//Math.PI / 2; //1

            /// <summary>
            /// Количество узлов на предыдущем слое
            /// </summary>
            public int inputLayerSize = 0;

            /// <summary>
            /// Вектор входных весов нейрона
            /// </summary>
            public double[] weights = null;

            /// <summary>
            /// Вес на сигнале поляризации
            /// </summary>
            public double biasWeight = 0.01;

            /// <summary>
            /// Ссылка на предыдущий слой нейронов 
            /// </summary>
            public Node[] inputLayer = null;

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
                return 1 / (1 + System.Math.Exp(-inp));//Math.Atan(inp);//1 / (1 + Math.Exp(-inp));
            }
        }

        public double LearningSpeed = 0.01;
        private Node[] Sensors;
        private Node[][] Layers;
        private Node[] Outputs;
        public Stopwatch stopWatch = new Stopwatch();

        public void ReInit(int[] structure, double initialLearningRate = 0.25)
        {
            Layers = new Node[structure.Length][];

            Layers[0] = new Node[structure[0]];
            for (int neuron = 0; neuron < structure[0]; ++neuron)
                Layers[0][neuron] = new Node(null);
            Sensors = Layers[0];

            for (int layer = 1; layer < structure.Length; ++layer)
            {
                Layers[layer] = new Node[structure[layer]];
                for (int neuron = 0; neuron < structure[layer]; ++neuron)
                    Layers[layer][neuron] = new Node(Layers[layer - 1]);
            }
            Outputs = Layers[Layers.Length - 1];
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
                Sensors[i].output = image.input[i];

            for (int i = 1; i < Layers.Length; i++)
                for (int j = 0; j < Layers[i].Length; j++)
                    Layers[i][j].Activate();

            image.Output = new double[Layers[Layers.Length - 1].Length];

            for (int i = 0; i < Layers[Layers.Length - 1].Length; i++)
                image.Output[i] = Layers[Layers.Length - 1][i].output;

            image.recognizedClass = image.ProcessPrediction(image.Output);
        }

        public double[] Run(double[] input)
        {
            for (int i = 0; i < input.Length; i++)
                Sensors[i].output = input[i];

            for (int i = 1; i < Layers.Length; i++)
            for (int j = 0; j < Layers[i].Length; j++)
                Layers[i][j].Activate();

            var output = new double[Layers[Layers.Length - 1].Length];

            for (int i = 0; i < Layers[Layers.Length - 1].Length; i++)
                output[i] = Layers[Layers.Length - 1][i].output;

            return output;
        }

        /// <summary>
        /// Обратное распространение ошибки
        /// </summary>
        private void BackProp(Sample image, double ita)
        {
            for (int i = 0; i < Layers[Layers.Length - 1].Length; i++)
                Layers[Layers.Length - 1][i].error = image.error[i];

            var board = Layers.Length - 1;
            Parallel.For(0, board, t =>
            {
            for (int i = Layers.Length - 1; i >= 0; --i)
                for (int j = 0; j < Layers[i].Length; ++j)
                    Layers[i][j].BackpropError(ita);
            });
        }

        /// <summary>
        /// Распознавание одного образа
        /// </summary>

        public FigureType Predict(Sample sample)
        {
            if (sample.input[25] > sample.input[20])
                return (FigureType)0;
            else
                return (FigureType)1;
        }

        /// <summary>
        /// Обучение одному заданному образу
        /// </summary>
        public override int Train(Sample sample, double acceptableError, bool parallel)
        {
            int iters = 0;
            while (iters < 100)
            {
                Run(sample);

                Debug.WriteLine(sample.ToString());
                Debug.WriteLine("Estimated error : " + sample.EstimatedError().ToString());


                if (sample.EstimatedError() < 0.2 && sample.Correct())
                {
                    Debug.WriteLine("Time " + iters.ToString());
                    return iters;
                }

                ++iters;
                BackProp(sample, LearningSpeed);
            }
            return iters;
        }

        /// <summary>
        /// Вектор выходных значений
        /// </summary>
        /// <returns></returns>
        public double[] getOutput()
        {
            return Outputs.Select(n => n.output).ToArray();
        }

        public override double TrainOnDataSet(SamplesSet samplesSet, int epochs_count, double acceptable_erorr, bool parallel = true)
        {
            double guessLevel = 0;
            int epochs_num = 1;
            while (epochs_num < epochs_count)
            {
                guessLevel = 0;
                for (int i = 0; i < samplesSet.samples.Count; i++)
                {
                    if (Train(samplesSet.samples.ElementAt(i), acceptable_erorr, parallel) == 0)
                        guessLevel += 1;
                }
                guessLevel /= samplesSet.samples.Count;
                if (guessLevel > acceptable_erorr)
                {
                    stopWatch.Stop();
                    return guessLevel;
                }
                epochs_num++;
            }
            return guessLevel;
        }

        public double TestOnDataSet(SamplesSet testSet)
        {
            if (testSet.Count == 0) return double.NaN;

            double guessLevel = 0;
            for (int i = 0; i < testSet.Count; ++i)
            {
                Sample s = testSet.samples.ElementAt(i);
                Predict(s);
                if (s.Correct()) guessLevel += 1;
            }
            return guessLevel / testSet.Count;
        }

        protected override double[] Compute(double[] input)
        {
            return Run(input);
        }
    }
}