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

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.timeseries.TimeSeriesData;
import ai.djl.timeseries.transform.TimeSeriesTransform;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.dataset.Record;

import java.util.List;

public abstract class TimeSeriesDataset extends RandomAccessDataset {

    protected List<TimeSeriesTransform> transformation;
    protected int contextLength;

    public static final FieldName[] DATASET_FIELD_NAMES = new FieldName[] {
        FieldName.TARGET,
        FieldName.FEAT_STATIC_CAT,
        FieldName.FEAT_STATIC_REAL,
        FieldName.FEAT_DYNAMIC_CAT,
        FieldName.FEAT_DYNAMIC_REAL
    };

    public TimeSeriesDataset(TimeSeriesBuilder<?> builder) {
        super(builder);
        transformation = builder.transformation;
        contextLength = builder.contextLength;
    }

    /** {@code TimeseriesDataset} override the get function so that it can preprocess the feature data as timeseries package way.
     * <p>{@inheritDoc} */
    @Override
    public Record get(NDManager manager, long index) {
        TimeSeriesData data = getTimeSeriesData(manager, index);

        data = apply(manager, data);
        if (!data.contains("PAST_" + FieldName.TARGET) || !data.contains("FUTURE_" + FieldName.TARGET)) {
            throw new IllegalArgumentException("Transformation must include InstanceSampler to split data into past and future part");
        }

        NDArray contextTarget = data.get("PAST_" + FieldName.TARGET).get("{}:", -contextLength + 1);
        NDArray futureTarget = data.get("FUTURE_" + FieldName.TARGET);
        NDList label = new NDList(contextTarget.concat(futureTarget, 0));

        return new Record(data.toNDList(), label);
    }

    public abstract TimeSeriesData getTimeSeriesData(NDManager manager, long index);

    /** Apply the preprocee transformation on {@link TimeSeriesData}. */
    private TimeSeriesData apply(NDManager manager, TimeSeriesData input) {
        for (TimeSeriesTransform transform : transformation) {
            input = transform.transform(manager, input, true);
        }
        return input;
    }

    public abstract static class TimeSeriesBuilder<T extends TimeSeriesBuilder<T>> extends RandomAccessDataset.BaseBuilder<T> {

        protected List<TimeSeriesTransform> transformation;
        protected int contextLength;

        public T setTransformation(List<TimeSeriesTransform> transformation) {
            this.transformation = transformation;
            return self();
        }

        public T setContextLength(int contextLength) {
            this.contextLength = contextLength;
            return self();
        }
    }
}
