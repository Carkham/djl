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

package ai.djl.timeseries.model.deepar;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.timeseries.distribution.Distribution;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

/** A deepar implements for prediction. */
public class DeepARPredictionNetwork extends DeepARNetwork {

    DeepARPredictionNetwork(Builder builder) {
        super(builder);
    }

    /** {@inheritDoc} */
    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        NDList unrollOutput = unrollLaggedRnn(parameterStore, inputs, training);

        NDArray repeatedScale = unrollOutput.get("scale").repeat(0,)
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return new Shape[0];
    }

    private Distribution outputDistribution(NDList params) {
        Distribution.DistributionBuilder<?> builder = distrOutput.distributionBuilder();
        builder.setDistrArgs(params);
        if (params.contains("scale")) {
            builder.optScale(params.get("scale"));
        }
        if (params.contains("loc")) {
            builder.optLoc(params.get("loc"));
        }

        return builder.build();
    }
}
