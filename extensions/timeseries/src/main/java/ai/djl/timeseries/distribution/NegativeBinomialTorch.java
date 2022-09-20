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

package ai.djl.timeseries.distribution;

import ai.djl.ndarray.NDArray;
import ai.djl.util.Preconditions;

public class NegativeBinomialTorch extends Distribution {

    private NDArray totalCount;
    private NDArray logits;

    public NegativeBinomialTorch(Builder builder) {
        totalCount = builder.distrArgs.get("total_count");
        logits = builder.distrArgs.get("logits");
    }

    @Override
    public NDArray logProb(NDArray target) {
        NDArray logUnnormalizedProb = totalCount.mul(logSigmoid(logits.mul(-1)))
                .add(target.mul(logSigmoid(logits)));

        NDArray logNormalization = totalCount.add(target).gammaln().mul(-1)
                .add(target.add(1).gammaln())
                .add(totalCount.gammaln());
        return logUnnormalizedProb.sub(logNormalization);
    }

    @Override
    public NDArray sample(int numSamples) {
        return null;
    }

    @Override
    public NDArray mean() {
        return totalCount.mul(logits.exp());
    }

    private NDArray logSigmoid(NDArray x) {
        return x.mul(-1).exp().add(1).getNDArrayInternal().rdiv(1).log();
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder extends DistributionBuilder<Builder> {
        /** {@inheritDoc} */
        @Override
        public Distribution build() {
            Preconditions.checkArgument(
                    distrArgs.contains("total_count"), "NegativeBinomial's args must contain mu.");
            Preconditions.checkArgument(
                    distrArgs.contains("logits"), "NegativeBinomial's args must contain alpha.");
            // We cannot scale using the affine transformation since negative binomial should return
            // integers. Instead we scale the parameters.
            if (scale != null) {
                NDArray logits = distrArgs.get("logits");
                logits.add(scale.log());
                logits.setName("logits");
                distrArgs.remove("logits");
                distrArgs.add(logits);
            }
            return new NegativeBinomialTorch(this);
        }

        /** {@inheritDoc} */
        @Override
        protected Builder self() {
            return this;
        }
    }
}
