/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.pytorch.zoo.cv.objectdetection;

import ai.djl.Application;
import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.FileTranslatorFactory;
import ai.djl.modality.cv.translator.InputStreamTranslatorFactory;
import ai.djl.modality.cv.translator.UrlTranslatorFactory;
import ai.djl.pytorch.zoo.PtModelZoo;
import ai.djl.repository.MRL;
import ai.djl.repository.Repository;
import ai.djl.repository.zoo.BaseModelLoader;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.Pipeline;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorFactory;
import ai.djl.util.Pair;
import ai.djl.util.Progress;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Path;
import java.util.Map;

/**
 * Model loader for Single Shot Detection models.
 *
 * <p>The model was trained on PyTorch and loaded in DJL in {@link
 * ai.djl.pytorch.engine.PtSymbolBlock}. See <a href="https://arxiv.org/pdf/1512.02325.pdf">SSD</a>.
 *
 * @see ai.djl.pytorch.engine.PtSymbolBlock
 */
public class SingleShotDetectionModelLoader
        extends BaseModelLoader<BufferedImage, DetectedObjects> {

    private static final Application APPLICATION = Application.CV.OBJECT_DETECTION;
    private static final String GROUP_ID = PtModelZoo.GROUP_ID;
    private static final String ARTIFACT_ID = "ssd";
    private static final String VERSION = "0.0.1";

    /**
     * Creates the Model loader from the given repository.
     *
     * @param repository the repository to load the model from
     */
    public SingleShotDetectionModelLoader(Repository repository) {
        super(repository, MRL.model(APPLICATION, GROUP_ID, ARTIFACT_ID), VERSION);
        FactoryImpl factory = new FactoryImpl();

        factories.put(new Pair<>(BufferedImage.class, DetectedObjects.class), factory);
        factories.put(
                new Pair<>(Path.class, DetectedObjects.class),
                new FileTranslatorFactory<>(factory));
        factories.put(
                new Pair<>(URL.class, DetectedObjects.class), new UrlTranslatorFactory<>(factory));
        factories.put(
                new Pair<>(InputStream.class, DetectedObjects.class),
                new InputStreamTranslatorFactory<>(factory));
    }

    /** {@inheritDoc} */
    @Override
    public Application getApplication() {
        return APPLICATION;
    }

    @Override
    public ZooModel<BufferedImage, DetectedObjects> loadModel(
            Map<String, String> filters, Device device, Progress progress)
            throws IOException, ModelNotFoundException, MalformedModelException {
        Criteria<BufferedImage, DetectedObjects> criteria =
                Criteria.builder()
                        .setTypes(BufferedImage.class, DetectedObjects.class)
                        .optFilters(filters)
                        .optDevice(device)
                        .optProgress(progress)
                        .build();
        return loadModel(criteria);
    }

    private static final class FactoryImpl
            implements TranslatorFactory<BufferedImage, DetectedObjects> {

        @Override
        public Translator<BufferedImage, DetectedObjects> newInstance(
                Map<String, Object> arguments) {
            int width = (Integer) arguments.getOrDefault("width", 300);
            int height = (Integer) arguments.getOrDefault("height", 300);
            double threshold = (Double) arguments.getOrDefault("threshold", 0.4d);
            int figSize = (Integer) arguments.getOrDefault("size", 300);
            int[] featSize =
                    (int[]) arguments.getOrDefault("feat_size", new int[] {38, 19, 10, 5, 3, 1});
            int[] steps =
                    (int[]) arguments.getOrDefault("steps", new int[] {8, 16, 32, 64, 100, 300});
            int[] scale =
                    (int[])
                            arguments.getOrDefault(
                                    "scale", new int[] {21, 45, 99, 153, 207, 261, 315});
            int[][] aspectRatio =
                    (int[][])
                            arguments.getOrDefault(
                                    "aspect_ratios",
                                    new int[][] {{2}, {2, 3}, {2, 3}, {2, 3}, {2}, {2}});

            Pipeline pipeline = new Pipeline();
            pipeline.add(new Resize(width, height))
                    .add(new ToTensor())
                    .add(
                            new Normalize(
                                    new float[] {0.485f, 0.456f, 0.406f},
                                    new float[] {0.229f, 0.224f, 0.225f}));

            return PtSSDTranslator.builder()
                    .setBoxes(figSize, featSize, steps, scale, aspectRatio)
                    .setPipeline(pipeline)
                    .setSynsetArtifactName("classes.txt")
                    .optThreshold((float) threshold)
                    .build();
        }
    }
}
