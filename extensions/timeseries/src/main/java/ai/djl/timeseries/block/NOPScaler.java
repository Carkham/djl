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

package ai.djl.timeseries.block;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

public class NOPScaler extends Scaler {

    NOPScaler(Builder builder) {
        super(builder);
    }

    /** {@inheritDoc} */
    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        NDArray data = inputs.get(0);
        NDArray scale = data.onesLike().mean(new int[] {dim}, keepDim);
        return new NDList(data, scale);
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder extends ScalerBuilder<Builder> {

        /** {@inheritDoc} */
        @Override
        protected Builder self() {
            return this;
        }

        /**
         * Return constructed {@code NOPScaler}.
         *
         * @return the constructed {@code NOPScaler}
         */
        public NOPScaler build() {
            validate();
            return new NOPScaler(this);
        }
    }
}
