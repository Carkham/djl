package ai.djl.timeseries.distribution;

import ai.djl.ndarray.NDArray;
import ai.djl.util.Preconditions;

public class StudentT extends Distribution {
    
    private NDArray mu;
    private NDArray sigma;
    private NDArray nu;

    StudentT(Builder builder) {
        mu = builder.distrArgs.get("mu");
        sigma = builder.distrArgs.get("sigma");
        nu = builder.distrArgs.get("nu");
    }

    @Override
    public NDArray logProb(NDArray target) {
        return null;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder extends DistributionBuilder<Builder> {

        @Override
        public Distribution build() {
            Preconditions.checkArgument(distrArgs.contains("mu"), "StudentTl's args must contain mu.");
            Preconditions.checkArgument(distrArgs.contains("sigma"), "StudentTl's args must contain sigma.");
            Preconditions.checkArgument(distrArgs.contains("nu"), "StudentTl's args must contain nu.");
            return new StudentT(this);
        }

        @Override
        protected Builder self() {
            return this;
        }
    }
}
