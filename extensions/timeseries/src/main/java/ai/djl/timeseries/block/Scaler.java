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

import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.util.Preconditions;

public abstract class Scaler extends AbstractBlock {

    private static final byte VERSION = 1;

    protected int dim;
    protected boolean keepDim;

    public Scaler(ScalerBuilder<?> builder) {
        super(VERSION);
        dim = builder.dim;
        keepDim = builder.keepDim;
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        Shape inputShape = inputShapes[0];
        Shape outputShape = new Shape();
        if (!this.keepDim) {
            for (int i = 0; i < inputShape.dimension(); i++) {
                if (i != this.dim) {
                    outputShape.add(inputShape.get(i));
                }
            }
        } else {
            outputShape.addAll(inputShape);
        }
        return new Shape[] {inputShape, outputShape};
    }

    public abstract static class ScalerBuilder<T extends ScalerBuilder> {

        protected int dim;
        protected boolean keepDim = false;

        /**
         * Set the dim to scale.
         *
         * @param dim which dim to scale
         * @return this Builder
         */
        public T setDim(int dim) {
            this.dim = dim;
            return self();
        }

        /**
         * Set whether to keep dim. Defaults to false;
         *
         * @param keepDim whether to keep dim
         * @return this Builder
         */
        public T optKeepDim(boolean keepDim) {
            this.keepDim = keepDim;
            return self();
        }

        /**
         * Validates that the required arguments are set.
         *
         * @throws IllegalArgumentException if the required arguments are illegal
         */
        protected void validate() {
            Preconditions.checkArgument(
                    dim > 0,
                    "Cannot compute scale along dim = 0 (batch dimension), please provide dim > 0");
        }

        protected abstract T self();
    }
}
