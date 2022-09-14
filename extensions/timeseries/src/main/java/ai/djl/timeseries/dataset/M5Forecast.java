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

import ai.djl.basicdataset.tabular.utils.Feature;
import ai.djl.basicdataset.tabular.utils.Featurizers;
import ai.djl.repository.MRL;
import ai.djl.repository.Repository;
import ai.djl.repository.zoo.DefaultModelZoo;
import ai.djl.util.JsonUtils;
import ai.djl.util.Progress;

import org.apache.commons.csv.CSVFormat;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * M5 Forecasting - Accuracy from <a
 * href="https://www.kaggle.com/competitions/m5-forecasting-accuracy">https://www.kaggle.com/competitions/m5-forecasting-accuracy</a>
 *
 * <p>
 */
public class M5Forecast extends CsvTimeSeriesDataset {

    private Usage usage;
    private MRL mrl;
    private boolean prepared;
    private Path root;
    private List<Integer> cardinality;

    /**
     * Creates a new instance of {@link M5Forecast} with the given necessary configurations.
     *
     * @param builder a builder with the necessary configurations
     */
    protected M5Forecast(Builder builder) {
        super(builder);
        usage = builder.usage;
        String path = builder.repository.getBaseUri().toString();
        mrl = MRL.undefined(builder.repository, DefaultModelZoo.GROUP_ID, path);
        root = Paths.get(mrl.getRepository().getBaseUri());
        cardinality = builder.cardinality;
    }

    /** {@inheritDoc} */
    @Override
    public void prepare(Progress progress) throws IOException {
        if (prepared) {
            return;
        }

        mrl.prepare(null, progress);
        Path csvFile = root.resolve(getUsagePath(usage));

        csvUrl = csvFile.toUri().toURL();
        super.prepare(progress);
        prepared = true;
    }

    /**
     * Return the cardinality of the dataset
     *
     * @return the cardinality of the dataset
     */
    public List<Integer> getCardinality() {
        return cardinality;
    }

    /**
     * Creates a builder to build a {@link M5Forecast}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    private String getUsagePath(Usage usage) {
        String usagePath;
        switch (usage) {
            case TRAIN:
                usagePath = "sales_train_validation.csv";
                return usagePath;
            case VALIDATION:
                usagePath = "sales_train_evaluation.csv";
                return usagePath;
            case TEST:
                usagePath = "sales_train_process.csv";
                return usagePath;
            default:
                throw new UnsupportedOperationException("Data not available.");
        }
    }

    public static class Builder extends CsvBuilder<Builder> {

        Repository repository;
        Usage usage = Usage.TRAIN;
        M5Features mf;
        List<Integer> cardinality;

        Builder() {
            csvFormat =
                    CSVFormat.DEFAULT
                            .builder()
                            .setHeader()
                            .setSkipHeaderRecord(true)
                            .setIgnoreHeaderCase(true)
                            .setTrim(true)
                            .build();
            cardinality = new ArrayList<>();
        }

        @Override
        protected Builder self() {
            return this;
        }

        public Builder setRepository(Repository repository) {
            this.repository = repository;
            return this;
        }

        public Builder optUsage(Usage usage) {
            this.usage = usage;
            return this;
        }

        public Builder addFeature(String name, FieldName fieldName) {
            return addFeature(name, fieldName, false);
        }

        public Builder addFeature(String name, FieldName fieldName, boolean onehotEncode) {
            parseFeatures();
            if (mf.categorical.contains(name)) {
                Map<String, Integer> map = mf.featureToMap.get(name);
                if (map == null) {
                    return addFieldFeature(
                            fieldName,
                            new Feature(name, Featurizers.getStringFeaturizer(onehotEncode)));
                }
                cardinality.add(map.size());
                return addFieldFeature(fieldName, new Feature(name, map, onehotEncode));
            }
            return addFieldFeature(fieldName, new Feature(name, true));
        }

        public M5Forecast build() {
            validate();
            return new M5Forecast(this);
        }

        private void parseFeatures() {
            if (mf == null) {
                try (InputStream is =
                                Objects.requireNonNull(
                                        M5Forecast.class.getResourceAsStream("m5forecast.json"));
                        Reader reader = new InputStreamReader(is, StandardCharsets.UTF_8)) {
                    mf = JsonUtils.GSON.fromJson(reader, M5Features.class);
                } catch (IOException e) {
                    throw new AssertionError("Failed to read m5forecast.json from classpath", e);
                }
            }
        }
    }

    private static final class M5Features {

        List<String> featureArray;
        Set<String> categorical;
        // categorical = String in featureArray its value indicate a String in featureToMap
        Map<String, Map<String, Integer>> featureToMap;
    }
}
