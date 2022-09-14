package ai.djl.timeseries.model.deepar;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

public final class DeepARTrainingNetwork extends DeepARNetwork {
    DeepARTrainingNetwork(Builder builder) {
        super(builder);
    }

    /** {@inheritDoc} */
    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        NDArray featStaticCat = inputs.get(0);
        NDArray featStaticReal = inputs.get(1);
        NDArray pastTimeFeat = inputs.get(2);
        NDArray pastTarget = inputs.get(3);
        NDArray pastObservedValues = inputs.get(4);
        //        NDArray pastIsPad = inputs.get(5);
        NDArray futureTimeFeat = inputs.get(6);
        NDArray futureTarget = inputs.get(7);
        NDArray futureObservedValues = inputs.get(8);

        NDList unrollOutput =
                unrollLaggedRnn(
                        parameterStore,
                        new NDList(
                                featStaticCat,
                                featStaticReal,
                                pastTimeFeat,
                                pastTarget,
                                pastObservedValues,
                                futureTimeFeat,
                                futureTarget),
                        training);

        NDArray ObservedValues =
                pastObservedValues
                        .get(":, {}:", -contextLength + 1)
                        .concat(futureObservedValues, 1);
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
