using Microsoft.ML.Data;

namespace NLP.TextFeaturizer.Models
{
    public class QueryPredictionModel
    {

        [ColumnName("PredictedLabel")]
        public uint PredictedLabel;

        [ColumnName("Label")]
        public uint Label;

        [ColumnName("Score")]
        public float[] Score;


    }

    public class QueryHintModel
    {
        [LoadColumn(0)]
        public uint Id { get; set; }

        [LoadColumn(1)]
        public string QueryHint { get; set; }

    }
}
