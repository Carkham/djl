package ai.djl.timeseries.evaluator;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.timeseries.distribution.output.DistributionOutput;
import ai.djl.training.evaluator.AbstractAccuracy;
import ai.djl.training.evaluator.Evaluator;
import ai.djl.training.loss.Loss;
import ai.djl.util.Pair;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class RMSSE extends Evaluator {

    private DistributionOutput distributionOutput;
    private int axis;
    private Map<String, Float> totalLoss;

    public RMSSE(DistributionOutput distributionOutput) {
        this("RMSSE", 1, distributionOutput);
    }

    public RMSSE(String name, int axis, DistributionOutput distributionOutput) {
        super(name);
        this.axis = axis;
        this.distributionOutput = distributionOutput;
        totalLoss = new ConcurrentHashMap<>();
    }
    
    protected Pair<Long, NDArray> accuracyHelper(NDList labels, NDList predictions) {
        NDArray label = labels.head();
        NDArray prediction = distributionOutput.distributionBuilder().setDistrArgs(predictions)
                .build().mean();

        checkLabelShapes(label, prediction);
        NDArray MeanSquare = label.sub(prediction).square().mean(new int[]{axis});
        NDArray scaleDenom = label.get(":, 1:").sub(label.get(":, :-1")).square().mean(new int[]{axis});
        
        NDArray rmsse = MeanSquare.div(scaleDenom).sqrt();
        rmsse = NDArrays.where(scaleDenom.eq(0), rmsse.onesLike(), rmsse);
        long total = rmsse.countNonzero().getLong();
        
        return new Pair<>(total, rmsse);
    }

    @Override
    public NDArray evaluate(NDList labels, NDList predictions) {
        return accuracyHelper(labels, predictions).getValue();
    }

    /** {@inheritDoc} */
    @Override
    public void addAccumulator(String key) {
        totalInstances.put(key, 0L);
        totalLoss.put(key, 0f);
    }

    /** {@inheritDoc} */
    @Override
    public void updateAccumulator(String key, NDList labels, NDList predictions) {
        Pair<Long, NDArray> update = accuracyHelper(labels, predictions);
        totalInstances.compute(key, (k, v) -> v + update.getKey());
        totalLoss.compute(key, (k, v) -> v + update.getValue().sum().getFloat());
    }

    /** {@inheritDoc} */
    @Override
    public void resetAccumulator(String key) {
        totalInstances.compute(key, (k, v) -> 0L);
        totalLoss.compute(key, (k, v) -> 0f);
    }

    /** {@inheritDoc} */
    @Override
    public float getAccumulator(String key) {
        Long total = totalInstances.get(key);
        if (total == null || total == 0) {
            return Float.NaN;
        }

        return (float) totalLoss.get(key) / totalInstances.get(key);
    }
}
