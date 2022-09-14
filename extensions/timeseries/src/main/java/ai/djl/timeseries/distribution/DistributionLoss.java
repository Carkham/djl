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
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.timeseries.distribution.output.DistributionOutput;
import ai.djl.training.loss.Loss;

public class DistributionLoss extends Loss {

    private DistributionOutput distrOutput;

    /**
     * Base class for metric with abstract update methods.
     *
     * @param name The display name of the Loss
     */
    public DistributionLoss(String name, DistributionOutput distrOutput) {
        super(name);
        this.distrOutput = distrOutput;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray evaluate(NDList labels, NDList predictions) {
        Distribution.DistributionBuilder<?> builder = distrOutput.distributionBuilder();
        builder.setDistrArgs(predictions);
        if (predictions.contains("scale")) {
            builder.optScale(predictions.get("scale"));
        }
        if (predictions.contains("loc")) {
            builder.optLoc(predictions.get("loc"));
        }

        NDArray loss = builder.build().logProb(labels.singletonOrThrow()).mul(-1);

        if (predictions.contains("loss_weights")) {
            NDArray lossWeights = predictions.get("loss_weights");
            NDArray weightedValue =
                    NDArrays.where(lossWeights.neq(0), loss.mul(lossWeights), loss.zerosLike());
            NDArray sumWeights = lossWeights.sum(new int[] {1}).maximum(1.);
            loss = weightedValue.sum(new int[] {1}).div(sumWeights);
            if (loss.isNaN().any().getBoolean()) {
                System.out.println("loss_weights");
                System.out.println(lossWeights.toDebugString(1000, 1000, 1000, 1000));
                System.out.println("weighted_value");
                System.out.println(weightedValue.toDebugString(1000, 1000, 1000, 1000));
                System.out.println("sum_weights");
                System.out.println(sumWeights.toDebugString(1000, 1000, 1000, 1000));
                System.out.println("loss");
                System.out.println(loss.toDebugString(1000, 1000, 1000, 1000));
            }
        }
        return loss;
    }
}
