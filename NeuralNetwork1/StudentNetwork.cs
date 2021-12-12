using System;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace NeuralNetwork1
{
    public class StudentNetwork : BaseNetwork
    {
        private readonly double[][,] _weights;
        private readonly double[][] _values;
        private readonly double[][] _errors;
        private const double Speed = 0.25;
        private readonly Stopwatch _stopWatch = new Stopwatch();

        public StudentNetwork(int[] structure, double lowerBound = -1, double upperBound = 1)
        {
            _values = new double[structure.Length][];
            _errors = new double[structure.Length][];
            for (int i = 0; i < structure.Length; i++)
            {
                _errors[i] = new double[structure[i]];
                _values[i] = new double[structure[i] + 1];
                _values[i][structure[i]] = 1;
            }
            _weights = new double[structure.Length - 1][,];
            for(int n = 0; n < structure.Length - 1; n++)
            {
                var rowsCount = structure[n]+1;
                var columnsCount = structure[n+1];
                _weights[n] = new double[rowsCount, columnsCount];
                var r = new Random();
                for (int i = 0; i < rowsCount; i++)
                    for (int j = 0; j < columnsCount; j++)
                        _weights[n][i, j] = r.NextDouble() * (upperBound - lowerBound) + lowerBound;
            }
        }

        private double Error(double[] output)
        {
            double result = 0;
            for (int i = 0; i < output.Length; i++) 
                result += Math.Pow(output[i] - _values[_values.Length - 1][i], 2);
            result /= output.Length;
            return result;
        }

        private void Run(double[] input)
        {
            for (int j = 0; j < input.Length; j++) 
                _values[0][j] = input[j];
            for (int i = 1; i < _values.GetLength(0); i++)
                MultiplyAndApplySigmoid(_values[i - 1], _weights[i - 1], _values[i]);
        }

        private static void MultiplyAndApplySigmoid(double[] vector, double [,] matrix, double[] result)
        {
            var rowsCount = matrix.GetLength(0);
            var colCount = matrix.GetLength(1);
            for (int i = 0; i < colCount ; i++)
            {
                double sum = 0;
                for(int j = 0; j < rowsCount; j++) 
                    sum += vector[j] * matrix[j, i];
                result[i] = Sigmoid(sum);
            }  
        }

        private static double Sigmoid(double value) => 1.0 / (Math.Exp(-value) + 1);

        private void BackPropagation(double[] output)
        {
            for (var j = 0; j < output.Length; j++)
            {
                var actualValue = _values[_errors.Length - 1][j];
                var expectedValue = output[j];
                _errors[_errors.Length - 1][j] = actualValue * (1 - actualValue) * (expectedValue - actualValue);
            }
            for (int i = _errors.Length - 2; i >= 1; i--)
                for (int j = 0; j < _errors[i].Length; j++)
                {
                    var value = _values[i][j];
                    value *= 1 - value;
                    var sum = 0.0;
                    for (int k = 0; k < _errors[i + 1].Length; k++)
                        sum += _errors[i + 1][k] * _weights[i][j, k];
                    _errors[i][j] = value * sum;
                }
        }

        private void ComputeWeights()
        {
           
            for(int n = 0; n < _weights.Length; n++)
            {
                for(int i = 0; i < _weights[n].GetLength(0); i++)
                {
                    for(int j = 0; j < _weights[n].GetLength(1); j++)
                    {
                        var dWeight = Speed * _errors[n + 1][j] * _values[n][i];
                        _weights[n][i, j] += dWeight;
                    }
                }
            }
        }

        public override int Train(Sample sample, double acceptableError, bool parallel)
        {
            int i = 1;
            while (sample.EstimatedError() > acceptableError)
            {
                Run(sample.input);
                BackPropagation(sample.Output);
                ComputeWeights();
                ++i;
            }

            return i;
        }

        public override double TrainOnDataSet(SamplesSet samplesSet, int epochsCount, double acceptableError, bool parallel)
        {
            //  Сначала надо сконструировать массивы входов и выходов
            double[][] inputs = new double[samplesSet.Count][];
            double[][] outputs = new double[samplesSet.Count][];

            //  Теперь массивы из samplesSet группируем в inputs и outputs
            for (int i = 0; i < samplesSet.Count; ++i)
            {
                inputs[i] = samplesSet[i].input;
                outputs[i] = samplesSet[i].Output;
            }

            //  Текущий счётчик эпох
            int epochToRun = 0;

            double error = double.PositiveInfinity;

#if DEBUG
            StreamWriter errorsFile = File.CreateText("errors.csv");
#endif

            _stopWatch.Restart();

            while (epochToRun++ < epochsCount && error > acceptableError)
            {
                //epochToRun++;
                error = 0;
                for (int i = 0; i < inputs.Length; i++)
                {
                    Run(inputs[i]);
                    BackPropagation(outputs[i]);
                    ComputeWeights();
                    error += Error(outputs[i]);
                }
                error /= inputs.Length;
                errorsFile.WriteLine(error);
                OnTrainProgress(epochToRun * 1.0 / epochsCount, error, _stopWatch.Elapsed);
            }

#if DEBUG
            errorsFile.Close();
#endif
            OnTrainProgress(epochToRun * 1.0 / epochsCount, error, _stopWatch.Elapsed);
            _stopWatch.Stop();
            return error;
        }

        protected override double[] Compute(double[] input)
        {
            Run(input);
            return _values.Last().Take(_values.Last().Length - 1).ToArray();
        }
    }
}