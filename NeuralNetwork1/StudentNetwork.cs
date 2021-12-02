using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork1
{
    public class StudentNetwork : BaseNetwork
    {
        /// <summary>
        /// Один нейрон сети
        /// </summary>
        private class Neuron
        {
            public double output = 0; //выходной сигнал
            public double error = 0; //ошибка
            public double[] inputWeights; //входные веса
            public Neuron[] prevLayer; //ссылка на предыдущий слой
            
            static Random random = new Random(); //генератор случайных чисел

            /// <summary>
            /// Создание нейрона первого слоя (сенсора)
            /// </summary>
            public Neuron() { }

            /// <summary>
            /// Создание нейронов второго и следующих слоев
            /// </summary>
            /// <param name="neurons">Ссылка на предыдущий слой</param>
            public Neuron(Neuron[] neurons)
            {
                prevLayer = neurons;
                inputWeights = new double[prevLayer.Length];
                //заполняем входные веса случайным образом от -1 до 1
                for (int i = 0; i < prevLayer.Length; i++)
                    inputWeights[i] = -1 + random.NextDouble() * 2;
            }

            /// <summary>
            /// Вычисление передаточной функции
            /// </summary>
            public void CalcGearFunction()
            {
                double weight = 0;
                for (int i = 0; i < prevLayer.Length; i++)
                    weight += prevLayer[i].output * inputWeights[i];

                output = 1 / (1 + Math.Exp(-weight)); //сигмоидальная передаточная функция
            }

            public void BackError()
            {
                //обрабатываем ошибку в текущем нейроне
                error *= output * (1 - output);

                //переносим ошибку на предыдущий слой
                for (int i = 0; i < prevLayer.Length; i++)
                    prevLayer[i].error += error * inputWeights[i];

                //корректируем веса
                for (int i = 0; i < prevLayer.Length; i++)
                    inputWeights[i] += error * prevLayer[i].output;

                error = 0;
            }
        }

        private List<List<Neuron>> neuronsLayers = new List<List<Neuron>>(); //список слоев сети

        public StudentNetwork(int[] structure)
        {
            for (int i = 0; i < structure.Length; i++)
            {
                List<Neuron> neurons = new List<Neuron>(); //создаем слой сети
                for (int j = 0; j < structure[i]; j++)
                    neurons.Add(i == 0 ? new Neuron() : new Neuron(neuronsLayers[i - 1].ToArray())); //создаем нейроны
                neuronsLayers.Add(neurons);
            }
        }

        /// <summary>
        /// Прямой проход
        /// </summary>
        private void Forward(Sample sample)
        {
            for (int i = 0; i < sample.input.Length; i++)
                neuronsLayers[0][i].output = sample.input[i]; //переносим значения на сенсоры

            //Выполняем послойно вычисления передаточных функций
            for (int i = 1; i < neuronsLayers.Count; i++)
                foreach (var neuron in neuronsLayers[i])
                    neuron.CalcGearFunction();

            for (int i = 0; i < neuronsLayers.Last().Count; i++)
                sample.Output[i] = neuronsLayers.Last()[i].output; //переносим значения с последнего слоя в вектор значений изображения

            sample.recognizedClass = Predict(sample);
        }

        /// <summary>
        /// Запускаем алгоритм обратного распространения ошибки
        /// </summary>
        void BackError(Sample sample)
        {
            //переносим ошибку из картинки на последний слой нейронов
            for (int i = 0; i < neuronsLayers.Last().Count; i++)
                neuronsLayers.Last()[i].error = sample.error[i];

            //переносим ошибки от слоя к слою
            for (int i = neuronsLayers.Count - 1; i >= 1; i--)
                foreach (var neuron in neuronsLayers[i])
                    neuron.BackError();
        }

        public override int Train(Sample sample, double acceptableError, bool parallel)
        {
            int cnt = 0;
            //запускаем сеть на примере до тех пор, пока правильно не распознается (максимум 100 раз)
            while (cnt < 100)
            {
                Forward(sample);

                if (sample.EstimatedError() < acceptableError && sample.Correct()) return cnt;

                //если мы здесь, значит ошибка распознавания (или слишком большая ошибка), запускаем алгоритм обратного распространения ошибки
                cnt++;
                BackError(sample);
            }
            return cnt;
        }

        public override double TrainOnDataSet(SamplesSet samplesSet, int epochsCount, double acceptableError, bool parallel)
        {
            var t = System.DateTime.Now;
            double accuracy = 0;

            int samplesLooked = 0; //сколько всего элементов было рассмотрено
            double allSamplesCount = samplesSet.samples.Count * epochsCount; //сколько всего элементов нужно будет рассмотреть

            while (epochsCount-- > 0)
            {
                double rightCnt = 0;
                //перебираем весь датасет
                foreach (var sample in samplesSet.samples)
                {
                    if (Train(sample, acceptableError, parallel) == 0) rightCnt++;
                    samplesLooked++;
                    if (samplesLooked % 25 == 0) //каждые 25 рассмотренных примеров обновляем индикатор прогресса
                        OnTrainProgress(samplesLooked / allSamplesCount, accuracy, DateTime.Now - t);
                }


                accuracy = rightCnt / samplesSet.samples.Count; //после конца каждой эпохи перессчитываем точность для передачи ее в форму
                if (accuracy >= 1 - acceptableError - 1e-10) //если точность соответствует допустимой ошибке, выходим
                {
                    OnTrainProgress(1, accuracy, DateTime.Now - t);
                    return accuracy;
                }

               OnTrainProgress(samplesLooked / allSamplesCount, accuracy, DateTime.Now - t);
            }

            OnTrainProgress(1, accuracy, DateTime.Now - t);
            return accuracy;
        }

        protected override double[] Compute(double[] input)
        {
            return neuronsLayers.Last().Select(neuron => neuron.output).ToArray();
        }
    }
}