"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const path_1 = __importDefault(require("path"));
const express_1 = __importDefault(require("express"));
const cors_1 = __importDefault(require("cors"));
const activation_functions_1 = require("./lib/neuronal-net/activation-functions");
const neuronal_net_1 = __importDefault(require("./lib/neuronal-net/neuronal-net"));
const dotenv = __importStar(require("dotenv"));
dotenv.config();
const app = (0, express_1.default)();
const PORT = process.env.BACKEND_PORT;
//  - - - - - MIDDLEWARE - - - - - 
app.use((0, cors_1.default)()); // allows your Vercel frontend to call this
app.use(express_1.default.json()); // built-in body parser (Express ≥ 4.16)
app.use(express_1.default.static(path_1.default.join(__dirname))); // serves index.html, styles.css, index.js …
//  Routes 
app.get('/', (_req, res) => {
    // res.sendFile(path.join(__dirname, 'index.html')); /* vercel now handles frontend */
    res.status(404).json({ message: 'Backend serves as API only. Frontend is on Vercel (😅 If I haven\'t changed it again.' });
});
// For Uptimerobot's interval ping check
app.get('/health', (req, res) => {
    res.status(200).json({ status: 'ok' });
});
/**
 * POST /run-network
 *
 * Receives the user's inputs and the network topology, builds a fresh network,
 * runs a forward pass, and returns the results.
 *
 * Body shape:
 * {
 *   input:        number[]   — values for each input neuron
 *   inputCount:   number     — number of input neurons
 *   hiddenLayers: number[]   — neurons per hidden layer, e.g. [4, 3]
 *   outputCount:  number     — number of output neurons
 * }
 */
app.post('/run-network', (req, res) => {
    const { input, inputCount, hiddenLayers, outputCount } = req.body;
    //  Basic validation 
    if (!Array.isArray(input) ||
        input.some((v) => typeof v !== 'number' || isNaN(v)) ||
        input.length !== inputCount) {
        res.status(400).json({ error: `Expected ${inputCount} numeric inputs.` });
        return;
    }
    try {
        const nn = new neuronal_net_1.default({
            activationFunction: activation_functions_1.EActivationFunction.SIGMOID,
            inputCount,
            hiddenLayers,
            outputCount,
        });
        const output = nn.send(input);
        res.json(output);
    }
    catch (err) {
        const message = err instanceof Error ? err.message : 'Unknown error';
        console.error('[NeuronalNet error]', message);
        res.status(400).json({ error: message });
    }
});
//  Start 
app.listen(PORT, () => {
    // console.log(`✓ ANN Visualizer running at http://localhost:${process.env.BACKEND_PORT}`);
    console.log(`✓ ANN API running on port ${PORT}`);
});
