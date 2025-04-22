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

const mongoUri = 'mongodb+srv://nwaozor:nwaozor@cluster0.rmvi7qm.mongodb.net/CASIDB?retryWrites=true&w=majority';
const client = new MongoClient(mongoUri);

let nlpManager = new NlpManager({ languages: ['en'], forceNER: true });
const learningEmitter = new EventEmitter();

async function loadPatternsAndTrainModels() {
    await client.connect();
    const db = client.db('CASIDB');
    const patterns = db.collection('patterns');
    const allPatterns = await patterns.find({}).toArray();

    nlpManager = new NlpManager({ languages: ['en'], forceNER: true });

    for (const pattern of allPatterns) {
        let text;
        try {
            text = zlib.gunzipSync(Buffer.from(pattern.content, 'base64')).toString();
        } catch {
            text = '';
        }
        if (text && pattern.concept) {
            nlpManager.addDocument('en', text, pattern.concept);
            nlpManager.addAnswer('en', pattern.concept, text);
        }
    }

    await nlpManager.train();
}

learningEmitter.on('newPattern', async (pattern) => {
    let text;
    try {
        text = zlib.gunzipSync(Buffer.from(pattern.content, 'base64')).toString();
    } catch {
        text = '';
    }
    if (text && pattern.concept) {
        nlpManager.addDocument('en', text, pattern.concept);
        nlpManager.addAnswer('en', pattern.concept, text);
        await nlpManager.train();
    }
});

function preprocessPrompt(prompt) {
    return prompt
        .replace(/039/g, "'")
        .replace(/[\u2013\u2014]/g, '-')
        .replace(/’/g, "'")
        .replace(/[^\w\s.,!?']/g, '')
        .replace(/\.\.+/g, '.')
        .trim()
        .toLowerCase();
}

function formatResponse(text, maxWords, mood) {
    text = text
        .replace(/\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s+\d{4}\b/g, '')
        .replace(/\n+/g, ' ')
        .replace(/\s+/g, ' ')
        .replace(/[^\w\s.,!?]/g, '')
        .trim();
    
    const doc = nlp(text);
    let sentences = doc.sentences().out('array').slice(0, 3);
    if (sentences.length === 0) sentences = ['Let’s explore this topic together.'];
    
    if (mood === 'enthusiastic') {
        sentences = sentences.map(s => s.replace(/[.!?]$/, '!'));
    } else if (mood === 'empathetic') {
        sentences = sentences.map(s => s.replace(/[.!?]$/, '.'));
    }
    
    let wordCount = 0;
    const selectedSentences = [];
    for (const sentence of sentences) {
        const words = sentence.split(' ');
        if (wordCount + words.length <= maxWords) {
            selectedSentences.push(sentence);
            wordCount += words.length;
        } else {
            break;
        }
    }
    
    let finalText = selectedSentences.join(' ');
    finalText = finalText.charAt(0).toUpperCase() + finalText.slice(1);
    
    if (!/[.!?]/.test(finalText.slice(-1))) finalText += '.';
    
    return finalText;
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

function synthesizeResponse(prompt, nlpResult, fallbackText) {
    let response = '';
    if (nlpResult.answer) {
        response = nlpResult.answer;
    } else if (fallbackText) {
        response = fallbackText;
    } else {
        response = "I'm not sure, but let's explore this together!";
    }

    if (nlpResult.sentiment && nlpResult.sentiment.vote === 'negative') {
        response += " I understand this might be tough. I'm here to help.";
    } else if (nlpResult.sentiment && nlpResult.sentiment.vote === 'positive') {
        response += " That's great to hear!";
    }

    if (!/[.!?]$/.test(response)) response += '.';
    response = response.charAt(0).toUpperCase() + response.slice(1);

    return response;
}

async function generateResponse({ prompt, diversityFactor = 0.5, depth = 10, breadth = 5, maxWords = 100, mood = 'neutral', patterns }) {
    try {
        if (!prompt || typeof prompt !== 'string' || prompt.trim().length < 1) {
            throw new Error('Prompt must be a non-empty string');
        }

        if (!nlpManager) {
            await loadPatternsAndTrainModels();
        }

        const nlpResult = await nlpManager.process('en', prompt);

        let fallbackText = '';
        if (patterns && patterns.length > 0) {
            const best = patterns.find(p =>
                (nlpResult.intent && p.concept === nlpResult.intent) ||
                (p.concept && prompt.toLowerCase().includes(p.concept.toLowerCase()))
            );
            if (best) {
                try {
                    fallbackText = zlib.gunzipSync(Buffer.from(best.content, 'base64')).toString();
                } catch {
                    fallbackText = '';
                }
            }
        }

        const outputText = synthesizeResponse(prompt, nlpResult, fallbackText);

        const confidence = (nlpResult.intent !== 'None' && nlpResult.intent) ? 0.98 : 0.9;
        const outputId = crypto.randomBytes(12).toString('hex');

        await client.connect();
        const db = client.db('CASIDB');
        const patternsCol = db.collection('patterns');
        await patternsCol.insertOne({
            concept: nlpResult.intent !== 'None' ? nlpResult.intent : prompt,
            content: zlib.gzipSync(outputText).toString('base64'),
            entities: nlpResult.entities || [],
            sentiment: nlpResult.sentiment ? nlpResult.sentiment.score : 0,
            updatedAt: new Date(),
            confidence,
            source: 'generated',
            title: `Response to: ${prompt.slice(0, 50)}`
        });

        learningEmitter.emit('newPattern', {
            concept: nlpResult.intent !== 'None' ? nlpResult.intent : prompt,
            content: zlib.gzipSync(outputText).toString('base64')
        });

        return { text: outputText, confidence, outputId };
    } catch (error) {
        console.error('[Generate] Generation error:', error.message);
        throw { message: `Generation failed: ${error.message}`, status: 500 };
    } finally {
        await client.close();
    }
}

module.exports = { generateResponse, loadPatternsAndTrainModels };