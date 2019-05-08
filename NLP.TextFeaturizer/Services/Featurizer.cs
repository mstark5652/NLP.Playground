using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;
using NLP.TextFeaturizer.Models;
using System;
using System.Linq;

namespace NLP.TextFeaturizer.Services
{

    public class Featurizer
    {



        public static TransformerChain<Microsoft.ML.Transforms.KeyToValueMappingTransformer> Train(string dataPath)
        {

            var mlContext = new MLContext(seed: 1);
            var reader = mlContext.Data.CreateTextLoader(
                hasHeader: true,
                //separatorChar: '|',
                columns: new[]
                {
                    new TextLoader.Column("Id", DataKind.UInt32, 0),
                    new TextLoader.Column("QueryHint", DataKind.String, 1)
                }
            );

            if (!System.IO.File.Exists(dataPath))
            {
                throw new System.IO.FileNotFoundException();
            }

            var data = reader.Load(dataPath);


            var queries = data.GetColumn<string>("QueryHint");

            Console.WriteLine("Starting training pipeline...");

            var pipeline = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "Label", inputColumnName: "Id")

                // Extract text features
                .Append(mlContext.Transforms.Text.FeaturizeText(outputColumnName: "TextFeatures", inputColumnName: "QueryHint"))

                // Normalize the message for later transforms
                .Append(mlContext.Transforms.Text.NormalizeText(outputColumnName: "NormalizedQuery", inputColumnName: "QueryHint", keepNumbers: true, keepPunctuations: false))


                //// bag of words
                //.Append(new WordBagEstimator(mlContext, "BagOfWords", "NormalizedQuery"))

                //// bag of trigrams, using hashes instead of dictionary indices.
                //.Append(new WordHashBagEstimator(mlContext, "BagOfTrigrams", "NormalizedQuery",
                //            ngramLength: 3, allLengths: false))

                // tokenized
                .Append(mlContext.Transforms.Text.TokenizeIntoWords(outputColumnName: "TokenizedQuery", inputColumnName: "NormalizedQuery"))

                // word embeddings
                .Append(mlContext.Transforms.Text.ApplyWordEmbedding(outputColumnName: "Embeddings", inputColumnName: "TokenizedQuery",
                        modelKind: WordEmbeddingEstimator.PretrainedModelKind.GloVe50D))

                // TODO: change algo

                .Append(mlContext.MulticlassClassification.Trainers.NaiveBayes(labelColumnName: "Label", featureColumnName: "Embeddings"))


                //.Append(mlContext.Transforms.Conversion.MapKeyToValue(("Id", "PredictedLabel")));
                .Append(mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName: "PredictedLabel"))
            ;

            var model = pipeline.Fit(data);
            var transformedData = model.Transform(data);

            var features = transformedData.GetColumn<float[]>("TextFeatures").Take(20).ToArray();
            //var normalized = transformedData.GetColumn<object[]>(mlContext, "NormalizedQuery").Take(10).ToArray();
            var tokenizedQueries = transformedData.GetColumn<string[]>("TokenizedQuery").Take(20).ToArray();
            //var unigrams = transformedData.GetColumn<float[]>(mlContext, "BagOfWords").Take(10).ToArray();
            //var ngrams = transformedData.GetColumn<float[]>(mlContext, "BagOfTrigrams").Take(10).ToArray();
            var embeddings = transformedData.GetColumn<float[]>("Embeddings").Take(20).ToArray();


            return model;
        }


        public static void Predict(TransformerChain<Microsoft.ML.Transforms.KeyToValueMappingTransformer> model, params string[] queries)
        {
            if (queries == null || queries.Count() == 0)
            {
                Console.WriteLine("No queries to predict.");
                return;
            }

            var mlContext = new MLContext(seed: 1);
            var engine = mlContext.Model.CreatePredictionEngine<QueryHintModel, QueryPredictionModel>(model);

            foreach (var query in queries)
            {
                var prediction = engine.Predict(new QueryHintModel { QueryHint = query });

                Console.WriteLine($"Input: {query}\n");
                Console.WriteLine($"Prediction: \n{Newtonsoft.Json.JsonConvert.SerializeObject(prediction, Newtonsoft.Json.Formatting.Indented)}\n");

                //var schema = engine.OutputSchema;

                //VBuffer<ReadOnlyMemory<char>> slotNames = default;
                //engine.OutputSchema[nameof(QueryPredictionModel.PredictedLabel)].GetSlotNames(ref slotNames);
                //var first = slotNames.GetItemOrDefault(0).ToString();



                //Console.WriteLine("Schema: {0}", schema);

            }
        }


    }
}
