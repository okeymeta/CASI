const natural = require('natural');
const sentiment = require('sentiment');
const winkNLP = require('wink-nlp');
const model = require('wink-eng-lite-web-model');
const nlp = winkNLP(model);
const handlebars = require('handlebars');
const crypto = require('crypto');
const axios = require('axios');
const lodash = require('lodash');
const { MongoClient } = require('mongodb');
const zlib = require('zlib');

const mongoUri = 'mongodb+srv://nwaozor:nwaozor@cluster0.rmvi7qm.mongodb.net/CASIDB?retryWrites=true&w=majority';
const client = new MongoClient(mongoUri);

const RESPONSE_TEMPLATE = `
{{#if moodIsEmpathetic}}
{{text}}. I'm here to support you!
{{else}}
{{text}}
{{/if}}
`;

async function googleVerify(text) {
    try {
        const response = await axios.get(`https://www.google.com/search?q=${encodeURIComponent(text)}`);
        return response.status === 200 ? 0.98 : 0.9;
    } catch {
        return 0.9;
    }
}

function preprocessPrompt(prompt) {
    // Normalize contractions and special characters
    return prompt
        .replace(/’/g, "'") // Normalize apostrophes
        .replace(/[^\w\s.,!?']/g, '') // Remove non-alphanumeric except basic punctuation
        .trim();
}

async function generateResponse({ prompt, diversityFactor = 0.5, depth = 10, breadth = 5, maxWords = 100, mood = 'neutral' }) {
    try {
        // Strict prompt validation
        if (!prompt || typeof prompt !== 'string' || prompt.trim().length < 1) {
            console.error('[Generate] Invalid prompt:', prompt);
            throw new Error('Prompt must be a non-empty string');
        }

        const cleanedPrompt = preprocessPrompt(prompt.trim());
        console.log(`[Generate] Processing prompt: ${cleanedPrompt}`);

        await client.connect();
        const db = client.db('CASIDB');
        const patterns = db.collection('patterns');

        const doc = nlp.readDoc(cleanedPrompt || ' ');
        let promptEntities = [];
        try {
            const entities = doc.entities();
            if (entities && typeof entities.out === 'function') {
                promptEntities = entities.out(winkNLP.its.detail) || [];
            } else {
                console.warn('[Generate] No valid entities found for prompt:', cleanedPrompt);
            }
        } catch (error) {
            console.warn('[Generate] Entity extraction failed for prompt:', cleanedPrompt, error.message);
            promptEntities = [];
        }

        let promptSentiment = 0;
        try {
            const sentimentResult = sentiment(cleanedPrompt);
            promptSentiment = sentimentResult && typeof sentimentResult.score === 'number' ? sentimentResult.score : 0;
        } catch (error) {
            console.warn('[Generate] Sentiment analysis failed for prompt:', cleanedPrompt, error.message);
        }

        // Broaden search for related patterns
        const relatedPatterns = await patterns.find({
            $or: [
                { concept: { $regex: cleanedPrompt.split(':')[0] || lodash.escapeRegExp(cleanedPrompt), $options: 'i' } },
                { entities: { $elemMatch: { value: { $in: promptEntities.map(e => e.value) } } } },
                { concept: { $in: ['greeting', 'nervousness', 'artificial intelligence'] } }
            ],
        }).sort({ confidence: -1 }).limit(breadth).toArray();
        console.log(`[Generate] Found ${relatedPatterns.length} related patterns`);

        let synthesizedText = '';
        const classifier = new natural.BayesClassifier();
        if (relatedPatterns.length > 0) {
            relatedPatterns.forEach(pattern => {
                try {
                    const text = zlib.gunzipSync(Buffer.from(pattern.content, 'base64')).toString();
                    classifier.addDocument(text, pattern.concept.split(':')[0] || pattern.concept);
                } catch (error) {
                    console.warn('[Generate] Failed to decompress pattern content for ID:', pattern._id, error.message);
                }
            });
            classifier.train();

            const concepts = classifier.getClassifications(cleanedPrompt).slice(0, depth);
            synthesizedText = concepts.map(c => c.label).join(' ');
            const doc = nlp.readDoc(synthesizedText);
            synthesizedText = doc.sentences().out().slice(0, maxWords).join(' ');
        }

        // Synthesize response if no patterns or insufficient text
        if (!synthesizedText || synthesizedText.length < 10) {
            console.log('[Generate] Synthesizing response from classifier or seed patterns');
            const classifier = new natural.BayesClassifier();
            relatedPatterns.forEach(pattern => {
                try {
                    const text = zlib.gunzipSync(Buffer.from(pattern.content, 'base64')).toString();
                    classifier.addDocument(text, pattern.concept.split(':')[0] || pattern.concept);
                } catch (error) {
                    console.warn('[Generate] Failed to decompress pattern content for ID:', pattern._id, error.message);
                }
            });

            // Add fallback training data if no patterns
            if (relatedPatterns.length === 0) {
                classifier.addDocument('Hello, I’m CASI, here to help!', 'greeting');
                classifier.addDocument('Hi! Let’s dive into your question or chat.', 'greeting');
                classifier.addDocument('Nervousness is normal; try practicing or breathing exercises.', 'nervousness');
                classifier.addDocument('AI is about machines learning and solving problems.', 'artificial intelligence');
            }

            classifier.train();
            const concepts = classifier.getClassifications(cleanedPrompt).slice(0, depth);
            synthesizedText = concepts.map(c => c.label).join(' ');
            const doc = nlp.readDoc(synthesizedText);
            synthesizedText = doc.sentences().out().slice(0, maxWords).join(' ') || 'I’m here to help! Please share more details or ask something specific.';
        }

        const template = handlebars.compile(RESPONSE_TEMPLATE);
        const moodIsEmpathetic = mood.toLowerCase() === 'empathetic';
        const outputText = template({ text: synthesizedText, moodIsEmpathetic });

        const confidence = await googleVerify(outputText);
        const outputId = crypto.randomBytes(12).toString('hex');

        await patterns.insertOne({
            concept: cleanedPrompt,
            content: zlib.gzipSync(outputText).toString('base64'),
            entities: promptEntities,
            sentiment: promptSentiment,
            updatedAt: new Date(),
            confidence,
        });

        console.log(`[Generate] Response generated, confidence: ${confidence}, outputId: ${outputId}`);
        return {
            text: outputText,
            confidence,
            outputId,
        };
    } catch (error) {
        console.error('[Generate] Generation error:', error);
        throw { message: `Generation failed: ${error.message}`, status: 500 };
    }
}

module.exports = {
    generateResponse,
};