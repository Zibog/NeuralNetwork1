using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork1
{
    public class StudentNetwork : BaseNetwork
    {
        /// <summary>
        /// Нейрон сети
        /// </summary>
        private class Neuron
        {
            // Выходной сигнал
            public double Output; 
            // Ошибка
            public double Error; 
            // Входные веса
            private double[] _inputWeights; 
            // Предыдущий слой нейронов
            private Neuron[] _prevLayer; 
            // Генератор случайных чисел
            private static Random _rand = new Random(); 

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
                // Заполнение входных весов случайным образом от -1 до 1
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

                // Сигмоидальная передаточная функция
                Output = 1 / (1 + Math.Exp(-weight)); 
            }

            public void ErrorBackPropagation()
            {
                // Обновление ошибки в текущем нейроне
                Error *= Output * (1 - Output);

                // Перенос ошибки на предыдущий слой
                for (int i = 0; i < _prevLayer.Length; i++)
                    _prevLayer[i].Error += Error * _inputWeights[i];

                // Корректировка весов
                for (int i = 0; i < _prevLayer.Length; i++)
                    _inputWeights[i] += Error * _prevLayer[i].Output;

                Error = 0;
            }
        }

        // Список слоев сети
        private List<Neuron[]> neuronsLayers = new List<Neuron[]>(); 

        public StudentNetwork(int[] structure)
        {
            for (int i = 0; i < structure.Length; i++)
            {
                // Слой сети
                var neurons = new Neuron[structure[i]]; 
                // Создаем нейроны
                for (int j = 0; j < structure[i]; j++)
                    neurons[j] = i == 0 ? new Neuron() : new Neuron(neuronsLayers[i - 1]); 
                neuronsLayers.Add(neurons);
            }
        }

        /// <summary>
        /// Прямой проход
        /// </summary>
        private void Forward(Sample sample)
        {
            // Перенос значения на входные нейроны
            for (int i = 0; i < sample.input.Length; i++)
                neuronsLayers[0][i].Output = sample.input[i]; 

            // Послойное вычисление передаточных функций
            for (int i = 1; i < neuronsLayers.Count; i++)
                foreach (var neuron in neuronsLayers[i])
                    neuron.Activate();

            // Перенос значений с последнего слоя в выход сети у образа
            for (int i = 0; i < neuronsLayers.Last().Length; i++)
                sample.Output[i] = neuronsLayers.Last()[i].Output; 

            // Обрабатываем перенесённые значения
            sample.recognizedClass = sample.ProcessPrediction(sample.Output);
        }

        /// <summary>
        /// Прямой проход
        /// </summary>
        private void Forward(double[] input)
        {
            // Перенос значения на входные нейроны
            for (int i = 0; i < input.Length; i++)
                neuronsLayers[0][i].Output = input[i]; 

            // Послойное вычисление передаточных функций
            for (int i = 1; i < neuronsLayers.Count; i++)
                foreach (var neuron in neuronsLayers[i])
                    neuron.Activate();
        }

        /// <summary>
        /// Алгоритм обратного распространения ошибки
        /// </summary>
        void ErrorBackPropagation(Sample sample)
        {
            // Перенос ошибки из образа на последний слой нейронов
            for (int i = 0; i < neuronsLayers.Last().Length; i++)
                neuronsLayers.Last()[i].Error = sample.error[i];

            // Распространение ошибки между слоями
            for (int i = neuronsLayers.Count - 1; i >= 1; i--)
                foreach (var neuron in neuronsLayers[i])
                    neuron.ErrorBackPropagation();
        }

        public override int Train(Sample sample, double acceptableError, bool parallel)
        {
            int i = 0;
            // Запуск сети до тех пор, пока образ не будет распознан (максимум 100 раз)
            while (i < 100)
            {
                Forward(sample);

                if (sample.EstimatedError() < acceptableError && sample.Correct()) 
                    return i;

                i++;
                // Ошибка распознавания или ошибка велика, запуск алгоритма обратного распространения ошибки
                ErrorBackPropagation(sample);
            }
            return i;
        }

        public override double TrainOnDataSet(SamplesSet samplesSet, int epochsCount, double acceptableError, bool parallel)
        {
            var startTime = DateTime.Now;
            // Точность распознавания
            double accuracy = 0;
            // Количество уже рассмотренных элементов
            var samplesLooked = 0; 
            // Количество элементов к рассмотрению
            double samplesCount = samplesSet.samples.Count * epochsCount; 

            while (epochsCount-- > 0)
            {
                // Количество верных распознаваний
                double correctAnswers = 0;
                
                foreach (var sample in samplesSet.samples)
                {
                    if (Train(sample, acceptableError, parallel) == 0) 
                        correctAnswers++;
                    samplesLooked++;
                    // Индикатор прогресса обновляется каждые 25 проходов
                    if (samplesLooked % 25 == 0)
                        OnTrainProgress(samplesLooked / samplesCount, accuracy, DateTime.Now - startTime);
                }

                // Пересчёт точности
                accuracy = correctAnswers / samplesSet.samples.Count; 
                // Если точность соответствует допустимой ошибке - выход
                if (accuracy >= 1 - acceptableError - 1e-10) 
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