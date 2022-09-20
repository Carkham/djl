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

package ai.djl.timeseries.distribution.output;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.timeseries.distribution.Distribution;
import ai.djl.timeseries.distribution.NegativeBinomialTorch;
import ai.djl.util.PairList;

public class NegativeBinomialTorchOutput extends DistributionOutput{

    public NegativeBinomialTorchOutput() {
        argsDim = new PairList<>(2);
        argsDim.add("total_count", 1);
        argsDim.add("logits", 1);
    }

    @Override
    public NDList domainMap(NDList arrays) {
        NDArray totalCount = arrays.get(0);
        NDArray logits = arrays.get(1);
        totalCount = totalCount.getNDArrayInternal().softPlus().squeeze(-1);
        logits = logits.squeeze(-1);
        totalCount.setName("total_count");
        logits.setName("logits");
        return new NDList(totalCount, logits);
    }

    @Override
    public Distribution.DistributionBuilder<?> distributionBuilder() {
        return NegativeBinomialTorch.builder();
    }
}
