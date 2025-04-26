const { pipeline, env } = require('@xenova/transformers');
const axios = require('axios');

// Configure environment for local model caching
env.allowLocalModels = true;
env.localModelPath = './models';
env.backends.onnx.wasm.numThreads = 1;

// Check network connectivity to Hugging Face
async function checkHuggingFaceAccess() {
    try {
        const response = await axios.head('https://huggingface.co/Xenova/all-MiniLM-L6-v2', { timeout: 5000 });
        console.log('Hugging Face accessible:', response.status);
        return true;
    } catch (error) {
        console.error('Cannot access Hugging Face:', error.message);
        return false;
    }
}

async function test() {
    try {
        // Verify network access
        const hasAccess = await checkHuggingFaceAccess();
        if (!hasAccess) {
            console.warn('No access to Hugging Face; ensure internet connection or use local models.');
        }

        console.log('Testing sentence encoder...');
        let encoder;
        try {
            encoder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', { 
                device: 'cpu',
                cache_dir: './models',
                quantized: true
            });
        } catch (error) {
            console.warn('Failed to load Xenova/all-MiniLM-L6-v2:', error.message);
            console.log('Falling back to Xenova/distilroberta-base...');
            encoder = await pipeline('feature-extraction', 'Xenova/distilroberta-base', { 
                device: 'cpu',
                cache_dir: './models',
                quantized: true
            });
        }
        const output = await encoder('Hello world', { pooling: 'mean', normalize: true });
        console.log('Embedding:', Array.from(output.data).slice(0, 5));

        console.log('Testing text generator...');
        let generator;
        try {
            generator = await pipeline('text-generation', 'Xenova/distilgpt2', { 
                device: 'cpu',
                cache_dir: './models',
                quantized: true
            });
        } catch (error) {
            console.warn('Failed to load Xenova/distilgpt2:', error.message);
            console.log('Falling back to Xenova/gpt2...');
            generator = await pipeline('text-generation', 'Xenova/gpt2', { 
                device: 'cpu',
                cache_dir: './models',
                quantized: true
            });
        }
        const genOutput = await generator('Hello world', { max_new_tokens: 20, do_sample: true });
        console.log('Generated:', genOutput[0].generated_text);
    } catch (error) {
        console.error('Test failed:', error.message);
    }
}

test();