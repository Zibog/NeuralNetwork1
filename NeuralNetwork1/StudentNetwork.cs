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
            public double Output; //выходной сигнал
            public double Error; //ошибка
            private double[] _inputWeights; //входные веса
            private Neuron[] _prevLayer; //ссылка на предыдущий слой
            
            private static Random _rand = new Random(); //генератор случайных чисел

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
                _prevLayer = neurons;
                _inputWeights = new double[_prevLayer.Length];
                //заполняем входные веса случайным образом от -1 до 1
                for (int i = 0; i < _prevLayer.Length; i++)
                    _inputWeights[i] = -1 + _rand.NextDouble() * 2;
            }

            /// <summary>
            /// Вычисление передаточной функции
            /// </summary>
            public void Activate()
            {
                double weight = 0;
                for (int i = 0; i < _prevLayer.Length; i++)
                    weight += _prevLayer[i].Output * _inputWeights[i];

                Output = 1 / (1 + Math.Exp(-weight)); //сигмоидальная передаточная функция
            }

            public void ErrorBackPropagation()
            {
                //обрабатываем ошибку в текущем нейроне
                Error *= Output * (1 - Output);

                //переносим ошибку на предыдущий слой
                for (int i = 0; i < _prevLayer.Length; i++)
                    _prevLayer[i].Error += Error * _inputWeights[i];

                //корректируем веса
                for (int i = 0; i < _prevLayer.Length; i++)
                    _inputWeights[i] += Error * _prevLayer[i].Output;

                Error = 0;
            }
        }

        private List<Neuron[]> neuronsLayers = new List<Neuron[]>(); //список слоев сети

        public StudentNetwork(int[] structure)
        {
            for (int i = 0; i < structure.Length; i++)
            {
                var neurons = new Neuron[structure[i]]; //создаем слой сети
                for (int j = 0; j < structure[i]; j++)
                    neurons[j] = i == 0 ? new Neuron() : new Neuron(neuronsLayers[i - 1]); //создаем нейроны
                neuronsLayers.Add(neurons);
            }
        }

        /// <summary>
        /// Прямой проход
        /// </summary>
        private void Forward(Sample sample)
        {
            for (int i = 0; i < sample.input.Length; i++)
                neuronsLayers[0][i].Output = sample.input[i]; //переносим значения на сенсоры

            //Выполняем послойно вычисления передаточных функций
            for (int i = 1; i < neuronsLayers.Count; i++)
                foreach (var neuron in neuronsLayers[i])
                    neuron.Activate();

            for (int i = 0; i < neuronsLayers.Last().Length; i++)
                sample.Output[i] = neuronsLayers.Last()[i].Output; //переносим значения с последнего слоя в вектор значений изображения

            //sample.recognizedClass = Predict(sample);
            sample.recognizedClass =
                sample.ProcessPrediction(sample.Output);
        }

        private void Forward(double[] input)
        {
            for (int i = 0; i < input.Length; i++)
                neuronsLayers[0][i].Output = input[i]; //переносим значения на сенсоры

            //Выполняем послойно вычисления передаточных функций
            for (int i = 1; i < neuronsLayers.Count; i++)
                foreach (var neuron in neuronsLayers[i])
                    neuron.Activate();
        }

        /// <summary>
        /// Алгоритм обратного распространения ошибки
        /// </summary>
        void ErrorBackPropagation(Sample sample)
        {
            //переносим ошибку из картинки на последний слой нейронов
            for (int i = 0; i < neuronsLayers.Last().Length; i++)
                neuronsLayers.Last()[i].Error = sample.error[i];

            //переносим ошибки от слоя к слою
            for (int i = neuronsLayers.Count - 1; i >= 1; i--)
                foreach (var neuron in neuronsLayers[i])
                    neuron.ErrorBackPropagation();
        }

        public override int Train(Sample sample, double acceptableError, bool parallel)
        {
            int i = 0;
            //запускаем сеть на примере до тех пор, пока правильно не распознается (максимум 100 раз)
            while (i < 100)
            {
                Forward(sample);

                if (sample.EstimatedError() < acceptableError && sample.Correct()) 
                    return i;

                //если мы здесь, значит ошибка распознавания (или слишком большая ошибка), запускаем алгоритм обратного распространения ошибки
                i++;
                ErrorBackPropagation(sample);
            }
            return i;
        }

        public override double TrainOnDataSet(SamplesSet samplesSet, int epochsCount, double acceptableError, bool parallel)
        {
            var startTime = DateTime.Now;
            double accuracy = 0;
            var samplesLooked = 0; //сколько всего элементов было рассмотрено
            double samplesCount = samplesSet.samples.Count * epochsCount; //сколько всего элементов нужно будет рассмотреть

            while (epochsCount-- > 0)
            {
                double correctAnswers = 0;
                //перебираем весь датасет
                foreach (var sample in samplesSet.samples)
                {
                    if (Train(sample, acceptableError, parallel) == 0) 
                        correctAnswers++;
                    samplesLooked++;
                    if (samplesLooked % 25 == 0) //каждые 25 рассмотренных примеров обновляем индикатор прогресса
                        OnTrainProgress(samplesLooked / samplesCount, accuracy, DateTime.Now - startTime);
                }


                accuracy = correctAnswers / samplesSet.samples.Count; //после конца каждой эпохи перессчитываем точность для передачи ее в форму
                if (accuracy >= 1 - acceptableError - 1e-10) //если точность соответствует допустимой ошибке, выходим
                {
                    OnTrainProgress(1, accuracy, DateTime.Now - startTime);
                    return accuracy;
                }

                OnTrainProgress(samplesLooked / samplesCount, accuracy, DateTime.Now - startTime);
            }

            OnTrainProgress(1, accuracy, DateTime.Now - startTime);
            return accuracy;
        }

        protected override double[] Compute(double[] input)
        {
            Forward(input);
            return neuronsLayers.Last().Select(neuron => neuron.Output).ToArray();
        }
    }
}