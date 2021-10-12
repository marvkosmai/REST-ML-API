using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace REST_ML_API.Controllers
{
    #region model input class
    public class ModelInput
    {
        [ColumnName(@"Label")]
        public string Label { get; set; }

        [ColumnName(@"ImageSource")]
        public string ImageSource { get; set; }

    }
    #endregion

    #region model output class
    public class ModelOutput
    {
        [ColumnName("PredictedLabel")]
        public string Prediction { get; set; }

        public float[] Score { get; set; }
    }
    #endregion

    [ApiController]
    [Route("/api/images/recognition")]
    public class ImageRecognitionController : Controller
    {
        public readonly Lazy<PredictionEngine<ModelInput, ModelOutput>> PredictionEngine = new Lazy<PredictionEngine<ModelInput, ModelOutput>>(() => CreatePredictionEngine(), true);

        [HttpPost]
        public Prediction Post([FromForm] IFormFile image)
        {
            var filePath = Path.GetTempFileName();

            using (var stream = System.IO.File.Create(filePath))
            {
                image.CopyToAsync(stream).Wait();
            }

            ModelInput sampleData = new ModelInput()
            {
                ImageSource = filePath,
            };

            var predEngine = PredictionEngine.Value;

            var predictionResult = predEngine.Predict(sampleData);

            return new Prediction() { Number = predictionResult.Prediction, Propability = String.Join(",", predictionResult.Score) };
        }

        private static PredictionEngine<ModelInput, ModelOutput> CreatePredictionEngine()
        {
            var mlContext = new MLContext();
            ITransformer mlModel = mlContext.Model.Load(Path.GetFullPath("MLModel.zip"), out var _);
            return mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(mlModel);
        }
    }

}
