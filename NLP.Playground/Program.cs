using NLP.TextFeaturizer.Services;
using System;

namespace NLP.Playground
{
    class Program
    {
        const string INPUT_QUERY = "Hello";


        static void Main(string[] args)
        {

            var dataFile = System.IO.Path.Combine("AppData", "ResponseSet.csv");

            var model = Featurizer.Train(dataFile);
            //Featurizer.Predict(model, INPUT_QUERY);

            Console.Write("\nFinished Training...");
            string _in = null;
            do
            {
                Console.Write("> ");
                _in = Console.ReadLine();

                Featurizer.Predict(model, _in);

            } while (!string.IsNullOrWhiteSpace(_in));

        }
    }
}
