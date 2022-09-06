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
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Block;
import ai.djl.nn.recurrent.LSTM;
import ai.djl.timeseries.block.FeatureEmbedder;
import ai.djl.timeseries.block.MeanScaler;
import ai.djl.timeseries.block.NOPScaler;
import ai.djl.timeseries.block.Scaler;
import ai.djl.timeseries.distribution.output.DistributionOutput;
import ai.djl.timeseries.distribution.output.StudentTOutput;
import ai.djl.timeseries.timefeature.Lag;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

import java.util.*;

public abstract class DeepARNetwork extends AbstractBlock {

    protected int historyLength;
    protected int contextLength;
    protected int predictionLength;
    protected DistributionOutput distrOutput;
    protected Block paramProj;
    protected int numParallelSamples;
    protected FeatureEmbedder embedder;
    protected Scaler scaler;
    protected int rnnInputSize;
    protected LSTM rnn;
    protected List<Integer> cardinality;
    protected List<Integer> embeddingDimension;
    protected int numLayers;
    protected int hiddenSize;
    protected float dropoutRate;
    protected List<Integer> lagsSeq;
    protected boolean scaling;

    DeepARNetwork(Builder builder) {
        predictionLength = builder.predictionLength;
        contextLength = builder.contextLength != 0 ? builder.contextLength : predictionLength;
        distrOutput = builder.distrOutput;
        paramProj = addChildBlock("param_proj", distrOutput.getArgsProj());
        if (builder.embeddingDimension != null || builder.cardinality == null) {
            embeddingDimension = builder.embeddingDimension;
        } else {
            embeddingDimension = new ArrayList<>();
            for (int cat : cardinality) {
                embeddingDimension.add(Math.min(50, (cat + 1) / 2));
            }
        }
        lagsSeq = builder.lagsSeq == null ? Lag.getLagsForFreq(builder.freq) : builder.lagsSeq;

        embedder =
                addChildBlock(
                        "feture_embedder",
                        FeatureEmbedder.builder()
                                .setCardinalities(cardinality)
                                .setEmbeddingDims(embeddingDimension)
                                .build());
        if (scaling) {
            scaler =
                    addChildBlock(
                            "scaler",
                            MeanScaler.builder()
                                    .setDim(1)
                                    .optKeepDim(true)
                                    .optMinimumScale(1e-10f)
                                    .build());
        } else {
            scaler =
                    addChildBlock("scaler", NOPScaler.builder().setDim(1).optKeepDim(true).build());
        }
        rnn =
                addChildBlock(
                        "rnn_lstm",
                        LSTM.builder()
                                .setNumLayers(numLayers)
                                .setStateSize(hiddenSize)
                                .optDropRate(dropoutRate)
                                .optBatchFirst(true)
                                .optReturnState(true)
                                .build());
    }

    protected NDList unrollLaggedRnn(ParameterStore ps, NDList inputs, boolean training) {
        NDArray featStaticCat = inputs.get(0);
        NDArray featStaticReal = inputs.get(1);
        NDArray pastTimeFeat = inputs.get(2);
        NDArray pastTarget = inputs.get(3);
        NDArray pastObservedValues = inputs.get(4);
        NDArray futureTimeFeat = inputs.get(5);
        NDArray futureTarget = inputs.get(6);

        NDArray context = pastTarget.get(":,{}:", -contextLength);
        NDArray observedContext = pastObservedValues.get(":,{}:", -contextLength);
        NDArray scale =
                scaler.forward(ps, new NDList(context, observedContext), training)
                        .get(1);

        NDArray priorSequence = pastTarget.get(":,:{}", -contextLength).div(scale);
        NDArray sequence =
                futureTarget != null
                        ? context.concat(futureTarget.get(":, :-1"), -1).div(scale)
                        : context.div(scale);

        NDArray embeddedCat =
                embedder.forward(ps, new NDList(featStaticCat), training)
                        .singletonOrThrow();
        NDArray staticFeat =
                NDArrays.concat(
                        new NDList(Arrays.asList(embeddedCat, featStaticReal, scale.log())), 1);
        NDArray expandedStaticFeat = staticFeat.expandDims(1).repeat(1, sequence.getShape().get(1));

        NDArray timeFeat =
                futureTimeFeat != null
                        ? pastTimeFeat
                                .get(":, {}:", -this.contextLength + 1)
                                .concat(futureTimeFeat, 1)
                        : pastTimeFeat.get(":, {}:", -this.contextLength + 1);

        NDArray features = expandedStaticFeat.concat(timeFeat, -1);
        NDArray lags = laggedSequenceValues(lagsSeq, priorSequence, sequence);

        NDArray rnnInput = lags.concat(features, -1);

        NDList outputs = rnn.forward(ps, new NDList(rnnInput), training);
        NDArray output = outputs.get(0);
        NDArray newState = outputs.get(1);

        NDArray params = paramProj.forward(ps, new NDList(output), training).singletonOrThrow();
        return new NDList(params, scale, output, staticFeat, newState);
    }

    protected NDArray laggedSequenceValues(List<Integer> indices, NDArray priorSequence, NDArray sequence) {
        if (Collections.max(indices) > (int) priorSequence.getShape().get(1)) {
            throw new IllegalArgumentException(
                    String.format("lags cannot go further than prior sequence length, found lag %d while prior sequence is only %d-long", Collections.max(indices), priorSequence.getShape().get(1)));
        }
        try (NDManager scope = NDManager.subManagerOf(priorSequence)) {
            scope.tempAttachAll(priorSequence, sequence);
            NDArray fullSequence = priorSequence.concat(sequence, 1);

            NDList lagsValues = new NDList(indices.size());
            for (int lagIndex : indices) {
                long begin = -lagIndex - sequence.getShape().get(1);
                long end = -lagIndex;
                lagsValues.add(
                        end < 0
                                ? sequence.get(":, {}:{}", begin, end)
                                : sequence.get(":, {}:", begin)
                );
            }

            NDArray lags = NDArrays.stack(lagsValues, -1);
            return scope.ret(lags.reshape(lags.getShape().get(0), lags.getShape().get(1), -1));
        }
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        protected String freq;
        protected int contextLength;
        protected int predictionLength;
        protected int numParallelSamples;
        protected DistributionOutput distrOutput = new StudentTOutput();
        protected List<Integer> cardinality;
        protected List<Integer> embeddingDimension;
        protected List<Integer> lagsSeq;

        public Builder setFreq(String freq) {
            this.freq = freq;
            return this;
        }

        public Builder setPredictionLength(int predictionLength) {
            this.predictionLength = predictionLength;
            return this;
        }

        public Builder setCardinality(List<Integer> cardinality) {
            this.cardinality = cardinality;
            return this;
        }

        public Builder optDistrOutput(DistributionOutput distrOutput) {
            this.distrOutput = distrOutput;
            return this;
        }

        public Builder optContextLength(int contextLength) {
            this.contextLength = contextLength;
            return this;
        }

        public Builder optNumParallelSamples(int numParallelSamples) {
            this.numParallelSamples = numParallelSamples;
            return this;
        }

        public Builder optEmbeddingDimension(List<Integer> embeddingDimension) {
            this.embeddingDimension = embeddingDimension;
            return this;
        }

        public Builder optLagsSeq(List<Integer> lagsSeq) {
            this.lagsSeq = lagsSeq;
            return this;
        }
    }
}
