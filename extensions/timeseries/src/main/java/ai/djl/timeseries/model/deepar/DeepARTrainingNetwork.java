package ai.djl.timeseries.model.deepar;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

public final class DeepARTrainingNetwork extends DeepARNetwork {
    DeepARTrainingNetwork(Builder builder) {
        super(builder);
    }

    /** {@inheritDoc} */
    @Override
    protected void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {

        Shape targetShape = inputShapes[3].slice(2);
        Shape contextShape = new Shape(1, contextLength).addAll(targetShape);
        scaler.initialize(manager, dataType, contextShape, contextShape);
        long scaleSize = scaler.getOutputShapes(new Shape[]{contextShape, contextShape})[0].get(1);

        embedder.initialize(manager, dataType, inputShapes[0]);
        long embeddedCatSize = embedder.getOutputShapes(new Shape[]{inputShapes[0]})[0].get(1);

        Shape inputShape = new Shape(1, contextLength * 2L - 1).addAll(targetShape);
        Shape lagsShape = inputShape.add(lagsSeq.size());
        long featSize = inputShapes[2].get(2) + embeddedCatSize + scaleSize;
        Shape rnnInputShape = lagsShape.slice(0, lagsShape.dimension() - 1).add(lagsShape.tail() + featSize);
        rnn.initialize(manager, dataType, rnnInputShape);

        Shape rnnOutShape = rnn.getOutputShapes(new Shape[]{rnnInputShape})[0];
        paramProj.initialize(manager, dataType, rnnOutShape);
    }

    /** {@inheritDoc} */
    @Override
    protected NDList forwardInternal(ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params) {
        NDArray featStaticCat = inputs.get(0);
        NDArray featStaticReal = inputs.get(1);
        NDArray pastTimeFeat = inputs.get(2);
        NDArray pastTarget = inputs.get(3);
        NDArray pastObservedValues = inputs.get(4);
        //        NDArray pastIsPad = inputs.get(5);
        NDArray futureTimeFeat = inputs.get(6);
        NDArray futureTarget = inputs.get(7);
        NDArray futureObservedValues = inputs.get(8);

        NDList unrollOutput = unrollLaggedRnn(parameterStore, new NDList(
                featStaticCat,
                featStaticReal,
                pastTimeFeat,
                pastTarget,
                pastObservedValues,
                futureTimeFeat,
                futureTarget
        ), training);

        NDArray target = pastTarget.get(":, {}:", -contextLength + 1).concat(futureTarget, 1);

        NDArray ObservedValues = pastObservedValues.get(":, {}:", -contextLength + 1).concat(futureObservedValues, 1);
        ObservedValues.setName("loss_weights");

        unrollOutput.add(ObservedValues);
        return unrollOutput;
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return new Shape[0];
    }
}
