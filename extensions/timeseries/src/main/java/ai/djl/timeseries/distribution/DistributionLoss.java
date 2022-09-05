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
import ai.djl.training.loss.Loss;

public abstract class DistributionLoss extends Loss {

    /**
     * Base class for metric with abstract update methods.
     *
     * @param name The display name of the Loss
     */
    public DistributionLoss(String name) {
        super(name);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray evaluate(NDList labels, NDList predictions) {
        NDArray observedValues = predictions.get()
        return null;
    }

    /**
     * Compute the log of the probability density/mass function evaluated at target
     *
     * @param target {@link NDArray} of shape (*batch_shape, *event_shape)
     * @param distrArgs the probability distribution args
     * @return Tensor of shape (batch_shape) containing the probability log-density for each event in target
     */
    public abstract NDList logProb(NDList target, NDList distrArgs);
}
