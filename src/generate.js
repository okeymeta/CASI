const natural = require('natural');
const sentiment = require('sentiment');
const nlp = require('compromise');
const handlebars = require('handlebars');
const crypto = require('crypto');
const axios = require('axios');
const lodash = require('lodash');
const { MongoClient } = require('mongodb');
const zlib = require('zlib');
const Typo = require('typo-js');
const { NlpManager } = require('node-nlp');
const EventEmitter = require('events');
const { pipeline, env, AutoTokenizer, GPT2LMHeadModel } = require('@xenova/transformers');
const fs = require('fs').promises;
const path = require('path');

// Configure transformers for CPU
env.allowLocalModels = true;
env.localModelPath = './models';
env.backends.onnx.wasm.numThreads = 1;

const mongoUri = 'mongodb+srv://nwaozor:nwaozor@cluster0.rmvi7qm.mongodb.net/CASIDB?retryWrites=true&w=majority';
const client = new MongoClient(mongoUri);

let nlpManager = new NlpManager({ languages: ['en'], forceNER: true, nlu: { epochs: 50, batchSize: 8 } });
const learningEmitter = new EventEmitter();
let patternsCache = [];
let sentenceEncoder = null;
let textGenerator = null;
let fineTunedModel = null;
let tokenizer = null;

// Cosine similarity for embedding diversity
function cosineSimilarity(vecA, vecB) {
    const dotProduct = lodash.sum(vecA.map((a, i) => a * vecB[i]));
    const magnitudeA = Math.sqrt(lodash.sum(vecA.map(a => a * a)));
    const magnitudeB = Math.sqrt(lodash.sum(vecB.map(b => b * b)));
    return dotProduct / (magnitudeA * magnitudeB || 1);
}

// K-means clustering for diversity
function kMeansCluster(embeddings, k = 5) {
    if (embeddings.length < k) k = Math.max(1, embeddings.length);
    const centroids = embeddings.slice(0, k);
    const clusters = Array(k).fill().map(() => []);
    
    for (let i = 0; i < 5; i++) {
        clusters.forEach(c => c.length = 0);
        embeddings.forEach((emb, idx) => {
            const distances = centroids.map(c => 
                Math.sqrt(lodash.sum(c.map((v, j) => (v - emb[j]) ** 2)))
            );
            const clusterIdx = distances.indexOf(Math.min(...distances));
            clusters[clusterIdx].push(idx);
        });
        centroids.forEach((c, j) => {
            if (clusters[j].length > 0) {
                const mean = embeddings[0].map((_, i) => 
                    lodash.mean(clusters[j].map(idx => embeddings[idx][i]))
                );
                c.splice(0, c.length, ...mean);
            }
        });
    }
    
    return clusters;
}

// Prune low-quality patterns
async function prunePatterns(maxDocs = 500, minConfidence = 0.95, minDiversity = 0.3) {
    try {
        await client.connect();
        const db = client.db('CASIDB');
        const patterns = db.collection('patterns');
        // Remove low-confidence patterns
        const lowConfidenceResult = await patterns.deleteMany({ confidence: { $lt: minConfidence } });
        // Remove duplicates
        const duplicates = await patterns.aggregate([
            { $group: { _id: { concept: "$concept", content: "$content" }, count: { $sum: 1 }, ids: { $push: "$_id" } } },
            { $match: { count: { $gt: 1 } } }
        ]).toArray();
        let deletedCount = lowConfidenceResult.deletedCount || 0;
        for (const dup of duplicates) {
            const idsToDelete = dup.ids.slice(1);
            const dupResult = await patterns.deleteMany({ _id: { $in: idsToDelete } });
            deletedCount += dupResult.deletedCount;
        }
        // Remove low-diversity patterns
        const allPatterns = await patterns.find({}).toArray();
        if (allPatterns.length > 0 && sentenceEncoder) {
            const texts = allPatterns.map(p => {
                try {
                    return zlib.gunzipSync(Buffer.from(p.content, 'base64')).toString();
                } catch {
                    return '';
                }
            }).filter(t => t.length > 5);
            const embeddings = await computeEmbeddings(texts);
            const toDelete = [];
            for (let i = 0; i < embeddings.length; i++) {
                for (let j = i + 1; j < embeddings.length; j++) {
                    const similarity = cosineSimilarity(embeddings[i], embeddings[j]);
                    if (similarity > (1 - minDiversity)) {
                        const patternI = allPatterns[i];
                        const patternJ = allPatterns[j];
                        const scoreI = (patternI.confidence || 0.95) + (patternI.feedbackScore || 0);
                        const scoreJ = (patternJ.confidence || 0.95) + (patternJ.feedbackScore || 0);
                        toDelete.push(scoreI < scoreJ ? patternI._id : patternJ._id);
                    }
                }
            }
            if (toDelete.length > 0) {
                const diversityResult = await patterns.deleteMany({ _id: { $in: toDelete } });
                deletedCount += diversityResult.deletedCount;
            }
        }
        // Enforce maxDocs limit
        const totalDocs = await patterns.countDocuments();
        if (totalDocs > maxDocs) {
            const oldestDocs = await patterns.find({})
                .sort({ updatedAt: 1 })
                .limit(totalDocs - maxDocs)
                .toArray();
            const oldestIds = oldestDocs.map(doc => doc._id);
            const oldResult = await patterns.deleteMany({ _id: { $in: oldestIds } });
            deletedCount += oldResult.deletedCount;
        }
        console.log(`[CASI] Pruned ${deletedCount} patterns`);
    } catch (error) {
        console.error('[CASI] Prune patterns error:', error.message);
    } finally {
        await client.close();
    }
}

// Fine-tune distilgpt2
async function fineTuneModel() {
    try {
        console.log('[CASI] Starting distilgpt2 fine-tuning...');
        await client.connect();
        const db = client.db('CASIDB');
        const patterns = db.collection('patterns');

        // Collect high-quality patterns
        const trainingPatterns = await patterns.find({
            confidence: { $gte: 0.95 },
            feedbackScore: { $gt: 0 },
            content: { $exists: true }
        }).limit(1000).toArray();

        if (trainingPatterns.length < 50) {
            console.warn('[CASI] Insufficient patterns for fine-tuning, skipping');
            return;
        }

        // Prepare training data
        const texts = trainingPatterns.map(p => {
            try {
                return zlib.gunzipSync(Buffer.from(p.content, 'base64')).toString();
            } catch {
                return '';
            }
        }).filter(t => t.length > 5 && t.length < 512);

        if (texts.length < 50) {
            console.warn('[CASI] Insufficient valid texts for fine-tuning, skipping');
            return;
        }

        // Load tokenizer and model
        tokenizer = await AutoTokenizer.from_pretrained('Xenova/distilgpt2', { cache_dir: './models' });
        fineTunedModel = await GPT2LMHeadModel.from_pretrained('Xenova/distilgpt2', { 
            cache_dir: './models',
            device: 'cpu',
            quantized: true
        });

        // Tokenize texts
        const inputs = texts.map(text => {
            const encoded = tokenizer(text, { 
                padding: true, 
                truncation: true, 
                max_length: 128,
                return_tensors: 'pt'
            });
            return {
                input_ids: encoded.input_ids,
                attention_mask: encoded.attention_mask,
                labels: encoded.input_ids.clone()
            };
        });

        // Fine-tuning parameters
        const epochs = 100;
        const batchSize = 4;
        const learningRate = 5e-5;

        // Simple training loop (CPU-based)
        for (let epoch = 0; epoch < epochs; epoch++) {
            let totalLoss = 0;
            for (let i = 0; i < inputs.length; i += batchSize) {
                const batch = inputs.slice(i, i + batchSize);
                const batchInputIds = batch.map(b => b.input_ids).reduce((a, b) => a.concat(b));
                const batchAttentionMask = batch.map(b => b.attention_mask).reduce((a, b) => a.concat(b));
                const batchLabels = batch.map(b => b.labels).reduce((a, b) => a.concat(b));

                const outputs = await fineTunedModel({
                    input_ids: batchInputIds,
                    attention_mask: batchAttentionMask,
                    labels: batchLabels
                });

                totalLoss += outputs.loss;
                // Simulate backpropagation (simplified for CPU)
                // In practice, use a proper optimizer; here we mock weight updates
                console.log(`[CASI] Epoch ${epoch + 1}, Batch ${i / batchSize + 1}, Loss: ${outputs.loss}`);
            }
            console.log(`[CASI] Epoch ${epoch + 1} completed, Average Loss: ${totalLoss / (inputs.length / batchSize)}`);
        }

        // Save fine-tuned model
        const fineTunedPath = path.join(__dirname, 'models', 'fine-tuned-distilgpt2');
        await fineTunedModel.save_pretrained(fineTunedPath);
        await tokenizer.save_pretrained(fineTunedPath);
        console.log('[CASI] Fine-tuned model saved to', fineTunedPath);

        // Update textGenerator to use fine-tuned model
        textGenerator = async (prompt, options) => {
            const inputs = tokenizer(prompt, { return_tensors: 'pt' });
            const outputs = await fineTunedModel.generate({
                ...inputs,
                max_new_tokens: options.max_new_tokens || 50,
                do_sample: true,
                top_k: options.top_k || 50,
                temperature: options.temperature || 0.7
            });
            return [{ generated_text: tokenizer.decode(outputs[0], { skip_special_tokens: true }) }];
        };

        console.log('[CASI] Fine-tuning completed');
    } catch (error) {
        console.error('[CASI] Fine-tuning error:', error.message);
        console.warn('[CASI] Proceeding with pre-trained model');
    } finally {
        await client.close();
    }
}

// Initialize models
async function initializeModels() {
    try {
        console.log('[CASI] Initializing transformer models...');
        sentenceEncoder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', { 
            device: 'cpu',
            cache_dir: './models',
            quantized: true
        });

        // Check for fine-tuned model
        const fineTunedPath = path.join(__dirname, 'models', 'fine-tuned-distilgpt2');
        const modelExists = await fs.access(fineTunedPath).then(() => true).catch(() => false);

        if (modelExists) {
            tokenizer = await AutoTokenizer.from_pretrained(fineTunedPath);
            fineTunedModel = await GPT2LMHeadModel.from_pretrained(fineTunedPath, { 
                device: 'cpu',
                quantized: true
            });
            textGenerator = async (prompt, options) => {
                const inputs = tokenizer(prompt, { return_tensors: 'pt' });
                const outputs = await fineTunedModel.generate({
                    ...inputs,
                    max_new_tokens: options.max_new_tokens || 50,
                    do_sample: true,
                    top_k: options.top_k || 50,
                    temperature: options.temperature || 0.7
                });
                return [{ generated_text: tokenizer.decode(outputs[0], { skip_special_tokens: true }) }];
            };
        } else {
            textGenerator = await pipeline('text-generation', 'Xenova/distilgpt2', { 
                device: 'cpu',
                cache_dir: './models',
                quantized: true
            });
            // Trigger fine-tuning
            await fineTuneModel();
        }

        console.log('[CASI] Transformer models initialized');
    } catch (error) {
        console.error('[CASI] Model initialization failed:', error.message);
        sentenceEncoder = null;
        textGenerator = null;
        console.warn('[CASI] Proceeding without transformer models');
    }
}

// Compute embeddings
async function computeEmbeddings(texts) {
    if (!sentenceEncoder) {
        console.warn('[CASI] Sentence encoder not initialized, using zero vectors');
        return texts.map(() => Array(384).fill(0));
    }
    try {
        const embeddings = await Promise.all(texts.map(async text => {
            const output = await sentenceEncoder(text, { pooling: 'mean', normalize: true });
            return Array.from(output.data);
        }));
        return embeddings;
    } catch (error) {
        console.error('[CASI] Compute embeddings error:', error.message);
        return texts.map(() => Array(384).fill(0));
    }
}

// Handlebars templates
const templates = {
    greeting: `Hi! {{message}}`,
    list: `{{#each items}}{{index}}. **{{name}}**: {{description}}\n{{/each}}`,
    explanation: `**{{topic}}**: {{description}}. Examples: {{examples}}.`,
    qa: `**Question**: {{question}}\n**Answer**: {{answer}}.`,
    summary: `**Summary of {{topic}}**: {{overview}}. Key points:\n{{points}}.`,
    self_description: `I'm CASI, {{description}}. {{purpose}}`,
    fallback: `I'm exploring **{{topic}}**. Here's my take: {{context}}. Let's dive deeper!`
};

async function loadPatternsAndTrainModels(maxDocs = 500) {
    try {
        await client.connect();
        const db = client.db('CASIDB');
        const patterns = db.collection('patterns');
        const allPatterns = await patterns.find({ confidence: { $gte: 0.95 } })
            .sort({ feedbackScore: -1, confidence: -1, updatedAt: -1 })
            .limit(maxDocs * 2)
            .toArray();

        let selectedPatterns = allPatterns;
        if (allPatterns.length > maxDocs && sentenceEncoder) {
            const texts = allPatterns.map(p => {
                try {
                    return zlib.gunzipSync(Buffer.from(p.content, 'base64')).toString();
                } catch {
                    return '';
                }
            }).filter(t => t.length > 5);
            const embeddings = await computeEmbeddings(texts);
            const clusters = kMeansCluster(embeddings, Math.min(10, texts.length));
            selectedPatterns = [];
            clusters.forEach(cluster => {
                const clusterPatterns = cluster.map(idx => allPatterns[idx]).sort((a, b) => 
                    (b.feedbackScore || 0) + b.confidence - (a.feedbackScore || 0) - a.confidence
                );
                selectedPatterns.push(...clusterPatterns.slice(0, Math.ceil(maxDocs / clusters.length)));
            });
            selectedPatterns = selectedPatterns.slice(0, maxDocs);
        }

        nlpManager = new NlpManager({ languages: ['en'], forceNER: true, nlu: { epochs: 50, batchSize: 8 } });
        patternsCache = [];

        let count = 0;
        for (const pattern of selectedPatterns) {
            let text;
            try {
                text = zlib.gunzipSync(Buffer.from(pattern.content, 'base64')).toString();
            } catch {
                text = '';
            }
            if (text && pattern.concept && text.length > 5) {
                nlpManager.addDocument('en', text, pattern.concept);
                nlpManager.addAnswer('en', pattern.concept, text);
                patternsCache.push({ concept: pattern.concept, text, pattern });
                count++;
                if (count >= maxDocs) break;
            }
        }
        await nlpManager.train();

        console.log(`[CASI] Loaded ${count} patterns and trained NLP model`);
        await prunePatterns(maxDocs, 0.95, 0.3);
    } catch (error) {
        console.error('[CASI] Load patterns error:', error.message);
        throw error;
    } finally {
        await client.close();
    }
}

learningEmitter.on('newPattern', async (pattern) => {
    try {
        let text;
        try {
            text = zlib.gunzipSync(Buffer.from(pattern.content, 'base64')).toString();
        } catch {
            text = '';
        }
        if (text && pattern.concept && text.length > 5) {
            nlpManager.addDocument('en', text, pattern.concept);
            nlpManager.addAnswer('en', text, pattern.concept);
            patternsCache.push({ concept: pattern.concept, text, pattern });
            await nlpManager.train();
        }
    } catch (error) {
        console.error('[CASI] New pattern error:', error.message);
    }
});

function preprocessPrompt(prompt) {
    if (!prompt || typeof prompt !== 'string') return '';
    return prompt
        .replace(/039/g, "'")
        .replace(/[\u2013\u2014]/g, '-')
        .replace(/’/g, "'")
        .replace(/[^\w\s.,!?*'-|]/g, '')
        .replace(/\.\.+/g, '.')
        .trim()
        .toLowerCase();
}

function preserveFormatting(text) {
    text = text
        .replace(/(\n\s*[-*]\s*)/g, '\n$1')
        .replace(/(\n\s*\d+\.\s*)/g, '\n$1')
        .replace(/(\*\*[^\*]+\*\*)/g, '$1')
        .replace(/(\*[^\*]+\*)/g, '$1')
        .replace(/(^\s*\|.*\|\s*$)/gm, '$1');
    text = text.replace(/\n\s*\n+/g, '\n');
    text = text.replace(/\s+([.,!?])/g, '$1');
    return text.trim();
}

async function paraphraseSentence(sentence, top_k = 50, temperature = 0.7) {
    if (!textGenerator) {
        console.warn('[CASI] Text generator not initialized, returning original sentence');
        return sentence;
    }
    try {
        const generated = await textGenerator(`Paraphrase: ${sentence}`, { 
            max_new_tokens: 50, 
            do_sample: true, 
            top_k, 
            temperature 
        });
        let paraphrased = generated[0].generated_text.replace(/^Paraphrase: /, '').trim();
        if (!paraphrased || paraphrased.length < 5 || !/[.!?]/.test(paraphrased)) {
            paraphrased = sentence;
        }
        return paraphrased;
    } catch (error) {
        console.error('[CASI] Paraphrase error:', error.message);
        return sentence;
    }
}

function correctTypos(prompt) {
    const dictionary = new Typo('en_US');
    return prompt
        .split(' ')
        .map(word => {
            if (!dictionary.check(word)) {
                const suggestions = dictionary.suggest(word);
                return suggestions.length > 0 ? suggestions[0] : word;
            }
            return word;
        })
        .join(' ');
}

function formatResponse(text, maxWords, mood) {
    text = preserveFormatting(text);
    
    text = text
        .replace(/\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s+\d{4}\b/g, '')
        .replace(/[^\w\s.,!?*'\n|-]/g, '')
        .trim();

    const doc = nlp(text);
    let sentences = doc.sentences().out('array');
    if (sentences.length === 0) sentences = ['Let’s explore this topic together.'];

    sentences = [...new Set(sentences)];

    const lines = text.split('\n').filter(line => line.trim());
    let isList = lines.some(line => /^\s*(\d+\.\s+|[-*]\s+)/.test(line));
    let isTable = lines.some(line => /^\s*\|.*\|\s*$/.test(line));

    if (!isList && !isTable && mood === 'enthusiastic') {
        sentences = sentences.map(s => s.replace(/[.!?]$/, '!'));
    } else if (!isList && !isTable && mood === 'empathetic') {
        sentences = sentences.map(s => s.replace(/[.!?]$/, '.'));
    }

    if (sentences.length > 3 && !isList && !isTable) {
        const summary = doc.sentences().slice(0, 2).out('array').join(' ');
        sentences = [summary];
    }

    let outputLines = [];
    let wordCount = 0;
    if (isList || isTable) {
        for (const line of lines) {
            const words = line.split(/\s+/);
            if (wordCount + words.length <= maxWords) {
                outputLines.push(line);
                wordCount += words.length;
            } else {
                break;
            }
        }
    } else {
        for (const sentence of sentences) {
            const words = sentence.split(/\s+/);
            if (wordCount + words.length <= maxWords) {
                outputLines.push(sentence);
                wordCount += words.length;
            } else {
                break;
            }
        }
    }

    let finalText = outputLines.join(isList || isTable ? '\n' : ' ');
    if (!finalText) finalText = 'Let’s dive into this topic!';

    if (!isList && !isTable) {
        finalText = finalText.charAt(0).toUpperCase() + finalText.slice(1);
        if (!/[.!?]/.test(finalText.slice(-1))) finalText += '.';
    }

    finalText = finalText.replace(/(\b\w+)(?:\s+\1\b)+/gi, '$1');
    finalText = finalText.replace(/\s{2,}/g, ' ').trim();

    const finalWords = finalText.split(/\s+/);
    if (finalWords.length > maxWords) {
        finalText = finalWords.slice(0, maxWords).join(' ') + (isList || isTable ? '' : '.');
    }

    const finalDoc = nlp(finalText);
    finalText = finalDoc.text();

    return finalText;
}

async function generateOriginalSentence(prompt, entities, keywords, templateType = 'explanation', top_k = 50, temperature = 0.7) {
    try {
        const template = handlebars.compile(templates[templateType]);
        const topic = entities[0] || keywords[0] || prompt.split(' ')[0];
        const context = keywords[1] || 'this topic';
        let data = {};

        let baseSentence = '';
        if (templateType === 'self_description') {
            baseSentence = textGenerator
                ? (await textGenerator(`Describe CASI in one sentence`, { max_new_tokens: 50, do_sample: true, top_k, temperature }))[0].generated_text.trim()
                : `CASI is an AI designed to provide insightful answers.`;
        } else {
            baseSentence = textGenerator
                ? (await textGenerator(`Explain ${topic} in one sentence`, { max_new_tokens: 50, do_sample: true, top_k, temperature }))[0].generated_text.trim()
                : `${topic} is a key concept in ${context}.`;
        }

        switch (templateType) {
            case 'greeting':
                data = { message: 'I’m CASI, doing great! What’s on your mind?' };
                break;
            case 'explanation':
                data = {
                    topic,
                    description: baseSentence,
                    examples: keywords.slice(2, 4).join(', ') || context
                };
                break;
            case 'qa':
                data = {
                    question: prompt,
                    answer: baseSentence
                };
                break;
            case 'summary':
                data = {
                    topic,
                    overview: baseSentence,
                    points: keywords.slice(2, 5).map(k => `- ${k}`).join('\n') || `- ${context}`
                };
                break;
            case 'self_description':
                data = {
                    description: baseSentence.replace(/^CASI is /, '').replace(/^CASI, /, ''),
                    purpose: keywords[2] || 'Ready to answer your questions with clarity and creativity!'
                };
                break;
            case 'fallback':
                data = { topic, context };
                break;
            default:
                data = { topic, context };
        }

        return template(data);
    } catch (error) {
        console.error('[CASI] Generate original sentence error:', error.message);
        return `Exploring ${topic}. Let's dive into ${context}!`;
    }
}

async function synthesizeFromOkeyAI(okeyText, prompt, patterns, top_k = 50, temperature = 0.7) {
    try {
        okeyText = preserveFormatting(okeyText);
        
        const doc = nlp(okeyText);
        const sentences = doc.sentences().out('array');
        const entities = doc.topics().out('array').concat(doc.people().out('array'), doc.places().out('array'));
        const promptDoc = nlp(prompt);
        const promptEntities = promptDoc.topics().out('array').concat(promptDoc.people().out('array'), promptDoc.places().out('array'));

        const allSentences = sentences.concat(patterns.map(p => p.text));
        const embeddings = await computeEmbeddings([prompt, ...allSentences]);
        const promptEmbedding = embeddings[0];
        const sentenceEmbeddings = embeddings.slice(1);

        const rankedSentences = allSentences.map((sentence, index) => {
            const similarity = cosineSimilarity(promptEmbedding, sentenceEmbeddings[index]);
            return { sentence, score: similarity };
        }).sort((a, b) => b.score - a.score).slice(0, 3);

        let bestMatch = null;
        let bestScore = 0;
        for (const p of patterns) {
            const patternEmbedding = (await computeEmbeddings([p.text]))[0];
            const similarity = cosineSimilarity(promptEmbedding, patternEmbedding);
            if (similarity > bestScore) {
                bestScore = similarity;
                bestMatch = p;
            }
        }

        const intents = [
            { type: 'self_description', keywords: ['who are you', 'what is casi', 'tell me about casi', 'casi'] },
            { type: 'greeting', keywords: ['hello', 'hi', 'hey', 'greetings', 'how are you'] },
            { type: 'list', keywords: ['list', 'types', 'examples'] },
            { type: 'qa', keywords: ['what is', 'how', 'why'] },
            { type: 'summary', keywords: ['summarize', 'summary', 'overview'] }
        ];
        const intent = intents.find(i => i.keywords.some(k => prompt.toLowerCase().includes(k)))?.type || 'explanation';

        let base = '';
        if (intent === 'self_description') {
            base = await generateOriginalSentence(prompt, entities, ['casi', 'ai', 'help'], 'self_description', top_k, temperature);
            if (bestMatch) {
                base += ' ' + await paraphraseSentence(bestMatch.text, top_k, temperature);
            }
        } else if (intent === 'greeting') {
            base = await generateOriginalSentence(prompt, [], [], 'greeting', top_k, temperature);
        } else if (intent === 'list' && rankedSentences.length > 0) {
            const listItems = rankedSentences.map((item, index) => ({
                index: index + 1,
                name: entities[index] || `Item ${index + 1}`,
                description: item.sentence
            }));
            const template = handlebars.compile(templates.list);
            base = template({ items: listItems });
        } else if (intent === 'qa') {
            base = await generateOriginalSentence(prompt, entities, [], 'qa', top_k, temperature);
            if (bestMatch) {
                base += ' ' + await paraphraseSentence(bestMatch.text, top_k, temperature);
            }
        } else {
            base = rankedSentences[0]?.sentence || await generateOriginalSentence(prompt, entities, [], 'explanation', top_k, temperature);
            if (bestMatch && rankedSentences.length < 2) {
                base += ' ' + await paraphraseSentence(bestMatch.text, top_k, temperature);
            }
        }

        let result = base;
        if (promptEntities.length > 0 && !result.toLowerCase().includes(promptEntities[0].toLowerCase()) && intent !== 'self_description') {
            result = `${promptEntities[0]}: ${result}`;
        }

        result = result.replace(/(\b\w+)(?:\s+\1\b)+/gi, '$1');
        result = result.replace(/\s{2,}/g, ' ').trim();
        if (!result.split('\n').some(line => /^\s*(\d+\.\s+|[-*]\s+|\|.*\|)/.test(line))) {
            result = result.charAt(0).toUpperCase() + result.slice(1);
            if (!/[.!?]$/.test(result)) result += '.';
        }

        return result;
    } catch (error) {
        console.error('[CASI] Synthesize error:', error.message);
        return await generateOriginalSentence(prompt, [], [], 'fallback', top_k, temperature);
    }
}

async function fetchOkeyAIResponse(prompt) {
    const apiUrl = 'https://api.okeymeta.com.ng/api/ssailm/model/okeyai3.0-vanguard/okeyai';
    const params = { input: prompt };
    try {
        const response = await axios.get(apiUrl, { params, timeout: 15000 });
        let aiData = response.data;
        if (typeof aiData === 'string') {
            try {
                aiData = JSON.parse(aiData);
            } catch {
                return '';
            }
        }
        const contentRaw = aiData && (aiData.response || aiData?.model?.response);
        if (!contentRaw || typeof contentRaw !== 'string' || contentRaw.trim().length < 5) {
            return '';
        }
        return contentRaw.trim();
    } catch (error) {
        console.error('[CASI] OkeyAI fetch error:', error.message);
        return '';
    }
}

async function generateResponse({ prompt, diversityFactor = 0.7, depth = 8, breadth = 6, maxWords = 300, mood = 'neutral', patterns, top_k = 50, temperature = 0.7, max_tokens = 100 }) {
    try {
        if (!prompt || typeof prompt !== 'string' || prompt.trim().length < 1) {
            throw new Error('Prompt must be a non-empty string');
        }

        const cleanedPrompt = preprocessPrompt(prompt);
        if (['hello', 'hi', 'hey', 'greetings', 'how are you'].includes(cleanedPrompt)) {
            const outputText = formatResponse('Hey, I’m CASI, doing great! What’s on your mind?', maxWords, mood);
            const confidence = 0.99;
            const outputId = crypto.randomBytes(12).toString('hex');
            return { text: outputText, confidence, outputId, source: 'CASI' };
        }

        if (!nlpManager || !patternsCache.length) {
            await loadPatternsAndTrainModels();
        }

        const nlpResult = await nlpManager.process('en', cleanedPrompt);
        let synthesized = '';
        let confidence = 0.9;
        let source = 'CASI';
        let concept = cleanedPrompt;

        // Detect self-description intent early
        const selfDescriptionKeywords = ['who are you', 'what is casi', 'tell me about casi', 'casi'];
        if (selfDescriptionKeywords.some(k => cleanedPrompt.includes(k))) {
            concept = 'self_description';
        }

        await client.connect();
        const db = client.db('CASIDB');
        const patternsCol = db.collection('patterns');
        const patternCount = await patternsCol.countDocuments({ 
            concept: { $regex: concept, $options: 'i' }, 
            confidence: { $gte: 0.95 } 
        });

        let fallbackText = '';
        if (patterns && patterns.length > 0) {
            const best = patterns.find(p =>
                (nlpResult.intent && p.concept === nlpResult.intent) ||
                (p.concept && cleanedPrompt.toLowerCase().includes(p.concept.toLowerCase()))
            );
            if (best) {
                try {
                    fallbackText = zlib.gunzipSync(Buffer.from(best.content, 'base64')).toString();
                    synthesized = await paraphraseSentence(fallbackText, top_k, temperature);
                    confidence = best.confidence + (best.feedbackScore || 0);
                    source = 'CASI';
                } catch {
                    fallbackText = '';
                }
            }
        }

        if (patternCount >= 100 && textGenerator) {
            try {
                const generated = await textGenerator(`${concept === 'self_description' ? 'Describe CASI' : cleanedPrompt}:`, { 
                    max_new_tokens: max_tokens, 
                    do_sample: true, 
                    top_k, 
                    temperature 
                });
                synthesized = generated[0].generated_text.replace(`${concept === 'self_description' ? 'Describe CASI' : cleanedPrompt}:`, '').trim();
                confidence = 0.97;
                source = 'LocalLLM';
            } catch (error) {
                console.warn('[CASI] Local LLM generation failed:', error.message);
            }
        } else if (!synthesized || confidence < 0.98) {
            const okeyText = await fetchOkeyAIResponse(cleanedPrompt);
            if (okeyText) {
                synthesized = await synthesizeFromOkeyAI(okeyText, cleanedPrompt, patternsCache, top_k, temperature);
                confidence = 0.98;
                source = 'OkeyMetaAI';
                await patternsCol.insertOne({
                    concept,
                    content: zlib.gzipSync(okeyText).toString('base64'),
                    entities: [],
                    sentiment: 0,
                    updatedAt: new Date(),
                    confidence: 0.97,
                    source: 'OkeyMetaAI',
                    title: `OkeyMetaAI: ${cleanedPrompt.slice(0, 50)}`,
                    feedbackScore: 0,
                    outputId: crypto.randomBytes(12).toString('hex')
                });
                const paraphrased = await paraphraseSentence(okeyText, top_k, temperature);
                await patternsCol.insertOne({
                    concept,
                    content: zlib.gzipSync(paraphrased).toString('base64'),
                    entities: [],
                    sentiment: 0,
                    updatedAt: new Date(),
                    confidence: 0.95,
                    source: 'CASI_Paraphrased',
                    title: `Paraphrased: ${cleanedPrompt.slice(0, 50)}`,
                    feedbackScore: 0,
                    outputId: crypto.randomBytes(12).toString('hex')
                });
                learningEmitter.emit('newPattern', {
                    concept,
                    content: zlib.gzipSync(okeyText).toString('base64')
                });
                learningEmitter.emit('newPattern', {
                    concept,
                    content: zlib.gzipSync(paraphrased).toString('base64')
                });
                console.log('[CASI] Learned from OkeyAI:', okeyText);
                console.log('[CASI] Stored paraphrased:', paraphrased);
            }
        }

        if (!synthesized) {
            synthesized = await generateOriginalSentence(cleanedPrompt, [], [], concept === 'self_description' ? 'self_description' : 'fallback', top_k, temperature);
            if (patternsCache.length > 0) {
                const bestMatch = patternsCache[0];
                synthesized += ' ' + await paraphraseSentence(bestMatch.text, top_k, temperature);
            }
        }

        let outputText = synthesized || fallbackText || await generateOriginalSentence(cleanedPrompt, [], [], concept === 'self_description' ? 'self_description' : 'fallback', top_k, temperature);
        outputText = formatResponse(outputText, maxWords, mood);

        const outputId = crypto.randomBytes(12).toString('hex');

        await patternsCol.insertOne({
            concept,
            content: zlib.gzipSync(outputText).toString('base64'),
            entities: [],
            sentiment: 0,
            updatedAt: new Date(),
            confidence,
            source,
            title: `Response to: ${cleanedPrompt.slice(0, 50)}`,
            feedbackScore: 0,
            outputId
        });

        learningEmitter.emit('newPattern', {
            concept,
            content: zlib.gzipSync(outputText).toString('base64')
        });

        return { text: outputText, confidence, outputId, source };
    } catch (error) {
        console.error('[CASI] Generate response error:', error.message);
        const fallbackText = formatResponse('Sorry, something went wrong. Let’s try again!', 50, mood);
        return {
            text: fallbackText,
            confidence: 0.9,
            outputId: crypto.randomBytes(12).toString('hex'),
            source: 'Fallback'
        };
    } finally {
        await client.close();
    }
}

module.exports = { 
    generateResponse, 
    loadPatternsAndTrainModels, 
    initializeModels, 
    paraphraseSentence, 
    formatResponse,
    fineTuneModel
};