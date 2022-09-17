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
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.timeseries.distribution.Distribution;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

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
        NDList state = new NDList(unrollOutput.get("hidden_state"), unrollOutput.get("cell_state"));
        String[] argNames = distrOutput.getArgsArray();
        NDList args = new NDList(argNames.length);
        for (String argName : distrOutput.getArgsArray()) {
            args.add(unrollOutput.get(argName));
        }

        NDArray repeatedScale = unrollOutput.get("scale").repeat(0, numParallelSamples);
        NDArray repeatedStaticFeat = unrollOutput.get("static_feat").repeat(0, numParallelSamples).expandDims(1);
        NDArray repeatedPastTarget = inputs.get(3).repeat(0, numParallelSamples).div(repeatedScale);
        NDArray repeatedTimeFeat = inputs.get(5).repeat(0, numParallelSamples);

        NDList repeatedState = new NDList(state.size());
        for (NDArray s : state) {
            repeatedState.add(s.repeat(0, numParallelSamples));
        }

        Distribution distr = outputDistribution(args, unrollOutput.get("scale"), 1);
        NDArray nextSample = distr.sample(numParallelSamples);
        NDList futureSamples = new NDList(predictionLength);
        futureSamples.add(nextSample);
        for (int k = 1; k < predictionLength; k++) {
            NDArray scaledNextSample = nextSample.div(repeatedScale);
            NDArray nextFeatures = repeatedStaticFeat.concat(repeatedTimeFeat.get(":, {}:{}", k, k+1), -1);
            NDArray nextLags = laggedSequenceValues(lagsSeq, repeatedPastTarget, scaledNextSample);
            NDArray rnnInput = nextLags.concat(nextFeatures, -1);

            NDList outputs = rnn.forward(parameterStore, new NDList(rnnInput).addAll(repeatedState), training);
            NDArray output = outputs.get(0);
            repeatedState = outputs.subNDList(1);

            repeatedPastTarget = repeatedPastTarget.concat(scaledNextSample, 1);

            args = paramProj.forward(parameterStore, new NDList(output), training);
            distr = outputDistribution(args, repeatedScale, 0);
            nextSample = distr.sample(numParallelSamples);
            futureSamples.add(nextSample);
        }

        NDArray futureSamplesConcat = NDArrays.concat(futureSamples, 1);
        return new NDList(futureSamplesConcat.reshape(-1, numParallelSamples, predictionLength));
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return new Shape[0];
    }

    private Distribution outputDistribution(NDList params, NDArray scale, int trailingN) {
        NDList slicedParams = params;
        if (trailingN > 0) {
            slicedParams = new NDList(params.size());
            for (NDArray p : params) {
                NDArray slicedP = p.get(":, {}:", -trailingN);
                slicedP.setName(p.getName());
            }
        }
        return distrOutput.distributionBuilder()
                .setDistrArgs(slicedParams)
                .optScale(scale)
                .build();
    }
}
