/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */

package ai.djl.examples.training;

import ai.djl.Model;
import ai.djl.basicdataset.tabular.utils.DynamicBuffer;
import ai.djl.basicdataset.tabular.utils.Feature;
import ai.djl.engine.Engine;
import ai.djl.examples.training.util.Arguments;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.repository.Repository;
import ai.djl.timeseries.Forecast;
import ai.djl.timeseries.dataset.FieldName;
import ai.djl.timeseries.dataset.M5Forecast;
import ai.djl.timeseries.distribution.DistributionLoss;
import ai.djl.timeseries.distribution.output.DistributionOutput;
import ai.djl.timeseries.distribution.output.NegativeBinomialOutput;
import ai.djl.timeseries.model.deepar.DeepARNetwork;
import ai.djl.timeseries.timefeature.TimeFeature;
import ai.djl.timeseries.transform.TimeSeriesTransform;
import ai.djl.timeseries.transform.convert.VstackFeatures;
import ai.djl.timeseries.transform.feature.AddAgeFeature;
import ai.djl.timeseries.transform.feature.AddObservedValuesIndicator;
import ai.djl.timeseries.transform.feature.AddTimeFeature;
import ai.djl.timeseries.transform.field.RemoveFields;
import ai.djl.timeseries.transform.field.SetField;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.SaveModelTrainingListener;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.util.ProgressBar;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Reader;
import java.net.URL;
import java.nio.FloatBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

public class TrainTimeSeries {

    private static String freq = "D";
    private static int predictionLength = 28;

    public static void main(String[] args) throws IOException {
        Repository repository = Repository.newInstance("test", Paths.get(
            System.getProperty("user.home") + "/Desktop/m5-forecasting-accuracy"
        ));

        Arguments arguments = new Arguments().parseArgs(args);
        try (Model model = Model.newInstance("deepar")) {
            DistributionOutput distributionOutput = new NegativeBinomialOutput();
            DefaultTrainingConfig config = setupTrainingConfig(arguments, distributionOutput);

            NDManager manager = model.getNDManager();
            DeepARNetwork trainingNetwork = getDeepARModel(distributionOutput);
            model.setBlock(trainingNetwork);
            List<TimeSeriesTransform> transformation = trainingNetwork.createTrainingTransformation(manager);
            int contextLength = trainingNetwork.getContextLength();

            RandomAccessDataset trainSet = getDataset(transformation, repository, contextLength);

            try {
                trainSet.get(manager, 0);
            } catch (IOException e) {
                e.printStackTrace();
            }

            try (Trainer trainer = model.newTrainer(config)) {
                trainer.setMetrics(new Metrics());

                Shape[] inputShapes = new Shape[9];
                inputShapes[0] = new Shape(1, 5);
                inputShapes[1] = new Shape(1, 1);

            }
        }
    }

    private static DefaultTrainingConfig setupTrainingConfig(Arguments arguments, DistributionOutput distributionOutput) {
        String outputDir = arguments.getOutputDir();
        SaveModelTrainingListener listener = new SaveModelTrainingListener(outputDir);
        listener.setSaveModelCallback(
                trainer -> {
                    TrainingResult result = trainer.getTrainingResult();
                    Model model = trainer.getModel();
                    float accuracy = result.getValidateEvaluation("Accuracy");
                    model.setProperty("Accuracy", String.format("%.5f", accuracy));
                    model.setProperty("Loss", String.format("%.5f", result.getValidateLoss()));
                });

        return new DefaultTrainingConfig(new DistributionLoss("neg_bionormal", distributionOutput))
                .addEvaluator(new Accuracy())
                .optDevices(Engine.getInstance().getDevices(arguments.getMaxGpus()))
                .addTrainingListeners(TrainingListener.Defaults.logging(outputDir))
                .addTrainingListeners(listener);
    }

    private static DeepARNetwork getDeepARModel(DistributionOutput distributionOutput) {
        List<Integer> cardinality = new ArrayList<>();
        cardinality.add(3);
        cardinality.add(10);
        cardinality.add(3);
        cardinality.add(7);
        cardinality.add(3049);

        return DeepARNetwork.builder()
            .setCardinality(cardinality)
            .setFreq(freq)
            .setPredictionLength(predictionLength)
            .optDistrOutput(distributionOutput)
            .optUseFeatStaticCat(true)
            .buildTrainingNetwork();
    }

    private static M5Forecast getDataset(List<TimeSeriesTransform> transformation, Repository repository, int contextLength) throws IOException {
        M5Forecast.Builder builder = M5Forecast.builder()
            .optUsage(Dataset.Usage.TRAIN)
            .setRepository(repository)
            .setTransformation(transformation)
            .setContextLength(contextLength)
            .setSampling(32, true);

        for (int i = 1; i <= 1913; i++) {
            builder.addFeature("d_" + i, FieldName.TARGET);
        }

        M5Forecast m5Forecast = builder.addFeature("state_id", FieldName.FEAT_STATIC_CAT)
            .addFeature("store_id", FieldName.FEAT_STATIC_CAT)
            .addFeature("cat_id", FieldName.FEAT_STATIC_CAT)
            .addFeature("dept_id", FieldName.FEAT_STATIC_CAT)
            .addFeature("item_id", FieldName.FEAT_STATIC_CAT)
            .build();
        m5Forecast.prepare(new ProgressBar());
        return m5Forecast;
    }
}
