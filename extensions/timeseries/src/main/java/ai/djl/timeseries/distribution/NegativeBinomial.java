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
import ai.djl.ndarray.NDManager;
import ai.djl.util.Preconditions;

public final class NegativeBinomial extends Distribution {

    private NDArray mu;
    private NDArray alpha;

    NegativeBinomial(Builder builder) {
        mu = builder.distrArgs.get("mu");
        alpha = builder.distrArgs.get("alpha");
    }

    @Override
    public NDArray logProb(NDArray target) {

        NDArray alphaInv = alpha.getNDArrayInternal().rdiv(1);
        NDArray alphaTimesMu = alpha.mul(mu);
        NDArray ret =
                target.mul(alphaTimesMu.div(alphaTimesMu.add(1)).log())
                        .sub(alphaInv.mul(alphaTimesMu.add(1).log()))
                        .add(target.add(alphaInv).gammaln())
                        .sub(target.add(1.).gammaln())
                        .sub(alphaInv.gammaln());
        if (ret.isNaN().any().getBoolean()) {
            System.out.println("alphaInv");
            System.out.println(alphaInv.toDebugString(10000, 1000, 1000, 1000));
            System.out.println("alpha");
            System.out.println(alpha.toDebugString(10000, 1000, 1000, 1000));
            System.out.println("mu");
            System.out.println(mu.toDebugString(10000, 1000, 1000, 1000));
            System.out.println("target");
            System.out.println(target.toDebugString(10000, 1000, 1000, 1000));
        }
        return ret;
    }

    @Override
    public NDArray sample(int numSamples) {
        NDManager manager = mu.getManager();
        NDArray expandedMu = mu.expandDims(0).repeat(0, numSamples);
        NDArray expandedAlpha = alpha.expandDims(0).repeat(0, numSamples);

        NDArray r = expandedAlpha.getNDArrayInternal().rdiv(1f);
        NDArray theta = expandedAlpha.mul(expandedMu);
        return manager.samplePoisson(manager.sampleGamma(r, theta));
    }

    @Override
    public NDArray mean() {
        return mu;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder extends DistributionBuilder<Builder> {

        @Override
        public Distribution build() {
            Preconditions.checkArgument(
                    distrArgs.contains("mu"), "NegativeBinomial's args must contain mu.");
            Preconditions.checkArgument(
                    distrArgs.contains("alpha"), "NegativeBinomial's args must contain alpha.");
            if (scale != null) {
                NDArray mu = distrArgs.get("mu");
                mu = mu.mul(scale);
                mu.setName("mu");
                distrArgs.remove("mu");
                distrArgs.add(mu);
            }
            return new NegativeBinomial(this);
        }

        @Override
        protected Builder self() {
            return this;
        }
    }
}
