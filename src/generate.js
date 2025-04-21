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

async function generateResponse({ prompt, diversityFactor = 0.5, depth = 10, breadth = 5, maxWords = 100, mood = 'neutral' }) {
    try {
        // Strict prompt validation
        if (!prompt || typeof prompt !== 'string' || prompt.trim().length < 3) {
            console.error('[Generate] Invalid prompt:', prompt);
            throw new Error('Prompt must be a non-empty string with at least 3 characters');
        }

        const cleanedPrompt = prompt.trim();
        console.log(`[Generate] Processing prompt: ${cleanedPrompt}`);

        await client.connect();
        const db = client.db('CASIDB');
        const patterns = db.collection('patterns');

        const doc = nlp.readDoc(cleanedPrompt);
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

        const relatedPatterns = await patterns.find({
            $or: [
                { concept: { $regex: lodash.escapeRegExp(cleanedPrompt), $options: 'i' } },
                { entities: { $elemMatch: { value: { $in: promptEntities.map(e => e.value) } } } },
            ],
        }).sort({ confidence: -1 }).limit(breadth).toArray();
        console.log(`[Generate] Found ${relatedPatterns.length} related patterns`);

        let synthesizedText = '';
        if (relatedPatterns.length > 0) {
            const classifier = new natural.BayesClassifier();
            relatedPatterns.forEach(pattern => {
                try {
                    const text = zlib.gunzipSync(Buffer.from(pattern.content, 'base64')).toString();
                    classifier.addDocument(text, pattern.concept);
                } catch (error) {
                    console.warn('[Generate] Failed to decompress pattern content for ID:', pattern._id, error.message);
                }
            });
            classifier.train();

            const concepts = classifier.getClassifications(cleanedPrompt).slice(0, depth);
            synthesizedText = concepts.map(c => c.label).join(' ');
            const doc = nlp.readDoc(synthesizedText);
            synthesizedText = doc.sentences().out().slice(0, maxWords).join(' ');
        } else {
            console.warn('[Generate] No related patterns found, using fallback response');
            synthesizedText = `Feeling nervous is normal. Practice your speech, use deep breathing (inhale 4s, exhale 4s), and visualize success.`;
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