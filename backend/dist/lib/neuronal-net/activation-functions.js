"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.ActivationFunction = exports.EActivationFunction = void 0;
var EActivationFunction;
(function (EActivationFunction) {
    EActivationFunction[EActivationFunction["SIGMOID"] = 0] = "SIGMOID";
})(EActivationFunction || (exports.EActivationFunction = EActivationFunction = {}));
class ActivationFunction {
    /** Sigmoid squashes any real number into the range (0, 1). */
    static sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }
    /**
     * Applies the chosen activation function and returns both the activation
     * value AND a "fired" flag for visualisation purposes.
     *
     * The raw sigmoid value is always forwarded to the next layer — un-fired
     * neurons still contribute their real output to downstream computation.
     */
    static activate(activationFunction, x) {
        switch (activationFunction) {
            case EActivationFunction.SIGMOID: {
                const value = ActivationFunction.sigmoid(x);
                const fired = value > 0.6 ? 1 : 0;
                return { value, fired, z: x };
            }
        }
    }
}
exports.ActivationFunction = ActivationFunction;
