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
import ai.djl.basicdataset.tabular.utils.Feature;
import ai.djl.engine.Engine;
import ai.djl.examples.training.util.Arguments;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.Repository;
import ai.djl.timeseries.dataset.FieldName;
import ai.djl.timeseries.dataset.M5Forecast;
import ai.djl.timeseries.dataset.TimeFeaturizers;
import ai.djl.timeseries.distribution.DistributionLoss;
import ai.djl.timeseries.distribution.output.DistributionOutput;
import ai.djl.timeseries.distribution.output.NegativeBinomialOutput;
import ai.djl.timeseries.model.deepar.DeepARNetwork;
import ai.djl.timeseries.transform.TimeSeriesTransform;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.dataset.Record;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.SaveModelTrainingListener;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;

import java.io.IOException;
import java.nio.file.Paths;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.*;

public class TrainTimeSeries {

    private static String freq = "D";
    private static int predictionLength = 28;

    public static void main(String[] args) throws IOException, TranslateException {
        TrainTimeSeries.runExample(args);
    }

    public static TrainingResult runExample(String[] args) throws IOException, TranslateException {
        Repository repository =
            Repository.newInstance(
                "test",
                Paths.get(
                    System.getProperty("user.home")
                        + "/Desktop/m5-forecasting-accuracy"));

        Arguments arguments = new Arguments().parseArgs(args);
        try (Model model = Model.newInstance("deepar")) {
            DistributionOutput distributionOutput = new NegativeBinomialOutput();
            DefaultTrainingConfig config = setupTrainingConfig(arguments, distributionOutput);

            NDManager manager = model.getNDManager();
            DeepARNetwork trainingNetwork = getDeepARModel(distributionOutput);
            model.setBlock(trainingNetwork);
            List<TimeSeriesTransform> transformation =
                trainingNetwork.createTrainingTransformation(manager);
            int contextLength = trainingNetwork.getContextLength();

            RandomAccessDataset trainSet = getDataset(transformation, repository, contextLength);

            try (Trainer trainer = model.newTrainer(config)) {
                trainer.setMetrics(new Metrics());

                Shape[] inputShapes = new Shape[9];
                inputShapes[0] = new Shape(1, 5);
                inputShapes[1] = new Shape(1, 1);
                inputShapes[2] = new Shape(1, 1121, 4);
                inputShapes[3] = new Shape(1, 1121);
                inputShapes[4] = new Shape(1, 1121);
                inputShapes[5] = new Shape(1, 1121);
                inputShapes[6] = new Shape(1, 28, 4);
                inputShapes[7] = new Shape(1, 28);
                inputShapes[8] = new Shape(1, 28);
                trainer.initialize(inputShapes);

                EasyTrain.fit(trainer, arguments.getEpoch(), trainSet, null);
                return trainer.getTrainingResult();
            }
        }
    }

    private static DefaultTrainingConfig setupTrainingConfig(
            Arguments arguments, DistributionOutput distributionOutput) {
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

    private static M5Forecast getDataset(
            List<TimeSeriesTransform> transformation, Repository repository, int contextLength)
            throws IOException {
        M5Forecast.Builder builder =
                M5Forecast.builder()
                        .optUsage(Dataset.Usage.TRAIN)
                        .setRepository(repository)
                        .setTransformation(transformation)
                        .setContextLength(contextLength)
                        .setSampling(32, true);

        for (int i = 1; i <= 1913; i++) {
            builder.addFeature("d_" + i, FieldName.TARGET);
        }

        M5Forecast m5Forecast =
                builder.addFeature("state_id", FieldName.FEAT_STATIC_CAT)
                        .addFeature("store_id", FieldName.FEAT_STATIC_CAT)
                        .addFeature("cat_id", FieldName.FEAT_STATIC_CAT)
                        .addFeature("dept_id", FieldName.FEAT_STATIC_CAT)
                        .addFeature("item_id", FieldName.FEAT_STATIC_CAT)
                        .addFieldFeature(
                                FieldName.START,
                                new Feature(
                                        "date",
                                        TimeFeaturizers.getConstantTimeFeaturizer(
                                                LocalDateTime.parse("2011-01-29T00:00"))))
                        .build();
        m5Forecast.prepare(new ProgressBar());
        return m5Forecast;
    }
}
