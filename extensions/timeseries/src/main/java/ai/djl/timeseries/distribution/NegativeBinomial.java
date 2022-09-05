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
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;

public final class NegativeBinomial extends Distribution {

    /**
     * Negative binomial distribution, i.e. the distribution of the number of
     * successes in a sequence of independent Bernoulli trials.
     *
     * @param mu Tensor containing the means, of shape (*batch_size, *event_shape).
     * @param alpha Tensor of the shape parameters, of shape (*batch_size, *event_shape).
     */
    public NegativeBinomial(NDArray mu, NDArray alpha) {
        eventShape = mu.getShape();
    }

    @Override
    public NDList logProb(NDList label, NDList distrArgs) {
        NDArray mu = distrArgs.get(0);
        NDArray alpha = distrArgs.get(1);
        NDArray target = label.singletonOrThrow();

        NDArray alphaInv = alpha.getNDArrayInternal().rdiv(1);
        NDArray alphaTimesMu = alpha.mul(mu);

        //TODO: add gammaln
        return new NDList(
            target.mul(alphaTimesMu.div(alphaTimesMu.add(1)))
            .sub(alphaInv.mul(alphaTimesMu.add(1).log()))
        );
    }

    @Override
    public NDArray resample(int numSamples) {
        return null;
    }
}
