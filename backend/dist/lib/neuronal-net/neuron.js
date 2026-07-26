"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const activation_functions_1 = require("./activation-functions");
/** Fixed bias added to the weighted sum before activation. */
const BIAS = 0.25;
class Neuron {
    constructor(options) {
        this.options = options;
        this.weights =
            options.weights ?? Neuron.randomWeights(options.previousLayerNeuronCount);
    }
    /** Xavier-style random initialisation in [-1, 1]. */
    static randomWeights(count) {
        return Array.from({ length: count }, () => Math.random() * 2 - 1);
    }
    /**
     * Computes the weighted sum of inputs plus bias.
     * Formula: Σ(xᵢ · wᵢ) + bias
     */
    weightedSum(inputs) {
        return (inputs.reduce((sum, x, i) => sum + x * this.weights[i], 0) + BIAS);
    }
    /**
     * Propagates inputs through this neuron.
     * @returns Activation result — the sigmoid value and whether the neuron fired.
     */
    send(inputs) {
        if (inputs.length !== this.weights.length) {
            throw new Error(`Input length (${inputs.length}) does not match weight count (${this.weights.length}).`);
        }
        const z = this.weightedSum(inputs);
        return activation_functions_1.ActivationFunction.activate(this.options.activationFunction, z);
    }
}
exports.default = Neuron;
