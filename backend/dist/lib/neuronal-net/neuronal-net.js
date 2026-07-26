"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const neuron_1 = __importDefault(require("./neuron"));
class NeuronalNet {
    constructor(options) {
        this.layers = NeuronalNet.buildLayers(options);
    }
    /**
     * Builds hidden + output layers.
     * Input layer neurons are implicit — they are not Neuron objects because
     * they simply pass values through without transformation.
     */
    static buildLayers(options) {
        const layerSizes = [...options.hiddenLayers, options.outputCount];
        return layerSizes.map((neuronCount, layerIndex) => {
            const prevCount = layerIndex === 0
                ? options.inputCount
                : layerSizes[layerIndex - 1];
            return Array.from({ length: neuronCount }, () => new neuron_1.default({
                activationFunction: options.activationFunction,
                previousLayerNeuronCount: prevCount,
            }));
        });
    }
    /**
     * Runs a forward pass through the network.
     * @param userInputs - Raw numeric inputs from the user (one per input neuron).
     */
    send(userInputs) {
        const eachLayerInputValues = [userInputs];
        const firedNeurons = [];
        const eachLayerZValues = [];
        const weights = [];
        let currentInputs = userInputs;
        for (const layer of this.layers) {
            const layerOutputs = [];
            const layerFired = [];
            const layerZ = [];
            const layerWeights = [];
            for (const neuron of layer) {
                const result = neuron.send(currentInputs);
                layerOutputs.push(result.value);
                layerFired.push(result.fired);
                layerZ.push(result.z);
                layerWeights.push(neuron.weights);
            }
            firedNeurons.push(layerFired);
            eachLayerZValues.push(layerZ);
            weights.push(layerWeights);
            currentInputs = layerOutputs;
            eachLayerInputValues.push(layerOutputs);
        }
        return {
            finalOutputs: currentInputs,
            firedNeurons,
            eachLayerInputValues,
            eachLayerZValues,
            weights,
        };
    }
}
exports.default = NeuronalNet;
