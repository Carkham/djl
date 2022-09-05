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
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Block;
import ai.djl.nn.recurrent.LSTM;
import ai.djl.nn.recurrent.RNN;
import ai.djl.timeseries.block.FeatureEmbedder;
import ai.djl.timeseries.block.MeanScaler;
import ai.djl.timeseries.block.NOPScaler;
import ai.djl.timeseries.block.Scaler;
import ai.djl.timeseries.timefeature.Lag;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

import java.util.ArrayList;
import java.util.List;

public class DeepARNetwork extends AbstractBlock {

    protected int historyLength;
    private int contextLength;
    private int predictionLength;
    private int numParallelSamples;
    private FeatureEmbedder embedder;
    private Scaler scaler;
    private int rnnInputSize;
    private LSTM rnn;
    private List<Integer> cardinality;
    private List<Integer> embeddingDimension;
    private int numLayers;
    private int hiddenSize;
    private float dropoutRate;
    private List<Integer> lagsSeq;
    private boolean scaling;

    DeepARNetwork(Builder builder) {
        predictionLength = builder.predictionLength;
        contextLength = builder.contextLength != 0 ? builder.contextLength : predictionLength;

        if (builder.embeddingDimension != null || builder.cardinality == null) {
            embeddingDimension = builder.embeddingDimension;
        } else {
            embeddingDimension = new ArrayList<>();
            for (int cat : cardinality) {
                embeddingDimension.add(Math.min(50, (cat + 1) / 2));
            }
        }
        lagsSeq = builder.lagsSeq == null ? Lag.getLagsForFreq()

        embedder = addChildBlock("feture_embedder", FeatureEmbedder.builder()
            .setCardinalities(cardinality)
            .setEmbeddingDims(embeddingDimension)
            .build());
        if (scaling) {
            scaler = addChildBlock("scaler", MeanScaler.builder()
                .setDim(1)
                .optKeepDim(true)
                .optMinimumScale(1e-10f)
                .build());
        } else {
            scaler = addChildBlock("scaler", NOPScaler.builder()
                .setDim(1)
                .optKeepDim(true)
                .build());
        }
        rnn = addChildBlock("rnn_lstm", LSTM.builder()
            .setNumLayers(numLayers)
            .setStateSize(hiddenSize)
            .optDropRate(dropoutRate)
            .optBatchFirst(true)
            .build());
    }


    @Override
    protected NDList forwardInternal(ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params) {
        return null;
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return new Shape[0];
    }

    private NDList unrollLaggedRnn(ParameterStore parameterStore, NDList input, boolean training) {
        NDArray featStaticCat = input.get(0);
        NDArray featStaticReal = input.get(1);
        NDArray pastTimeFeat = input.get(2);
        NDArray pastTarget = input.get(3);
        NDArray pastObservedValues = input.get(4);
        NDArray futureTimeFeat = input.get(5);
        NDArray futureTarget = input.get(6);

        NDArray context = pastTarget.get(":,{}:", -contextLength);
        NDArray observedContext = pastObservedValues.get(":,{}:", -contextLength);
        NDArray scale = scaler.forward(parameterStore, new NDList(context, observedContext), training).get(1);

        NDArray priorInput = pastTarget.get(":,:{}", -contextLength).div(scale);
        NDArray sequence;
        rnn.
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        private String freq;
        private int contextLength;
        private int predictionLength;
        private int numParallelSamples;
        private List<Integer> cardinality;
        private List<Integer> embeddingDimension;
        private List<Integer> lagsSeq;

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
