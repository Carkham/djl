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

package ai.djl.timeseries.dataset;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Blocks;
import ai.djl.nn.Parameter;
import ai.djl.repository.Repository;
import ai.djl.timeseries.transform.TimeSeriesTransform;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.Record;
import ai.djl.training.initializer.Initializer;
import ai.djl.training.loss.Loss;
import ai.djl.translate.TranslateException;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.nio.file.Paths;

public class M5ForecastTest {

    @Test
    public void testM5Forecast() throws IOException, TranslateException {
        TrainingConfig config =
                new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                        .optInitializer(Initializer.ONES, Parameter.Type.WEIGHT);

        Repository repository =
                Repository.newInstance(
                        "test",
                        Paths.get(
                                System.getProperty("user.home")
                                        + "/Desktop/m5-forecasting-accuracy"));

        try (Model model = Model.newInstance("model")) {
            model.setBlock(Blocks.identityBlock());

            NDManager manager = model.getNDManager();
            M5Forecast.Builder builder =
                    M5Forecast.builder()
                            .optUsage(Dataset.Usage.VALIDATION)
                            .setRepository(repository)
                            .setTransformation(TimeSeriesTransform.identityTransformation())
                            .setContextLength(7)
                            .setSampling(32, true);
            for (int i = 1; i <= 1941; i++) {
                builder.addFeature("d_" + i, FieldName.TARGET);
            }
            M5Forecast m5Forecast =
                    builder.addFeature("state_id", FieldName.FEAT_STATIC_CAT)
                            .addFeature("store_id", FieldName.FEAT_STATIC_CAT)
                            .addFeature("cat_id", FieldName.FEAT_STATIC_CAT)
                            .addFeature("dept_id", FieldName.FEAT_STATIC_CAT)
                            .addFeature("item_id", FieldName.FEAT_STATIC_CAT)
                            .build();

            m5Forecast.prepare();

            long size = m5Forecast.size();
            Assert.assertEquals(size, 30490);

            Record record = m5Forecast.get(manager, 0);
            NDList data = record.getData();
            NDList label = record.getLabels();
            NDArray featStatCat = data.get(1);
            Assert.assertEquals(featStatCat.toFloatArray(), new float[] {0f, 0f, 1f, 3f, 1437f});
            Assert.assertEquals(label.head().toFloatArray(), new float[] {0f, 0f, 3f, 3f, 0f, 1f});

            try (Trainer trainer = model.newTrainer(config)) {
                Batch batch = trainer.iterateDataset(m5Forecast).iterator().next();
                Assert.assertEquals(batch.getData().size(), 4);
                Assert.assertEquals(batch.getLabels().size(), 1);
                batch.close();
            }
        }
    }
}