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
const TfIdf = require('node-tfidf');

const mongoUri = 'mongodb+srv://nwaozor:nwaozor@cluster0.rmvi7qm.mongodb.net/CASIDB?retryWrites=true&w=majority';
const client = new MongoClient(mongoUri);

let nlpManager = new NlpManager({ languages: ['en'], forceNER: true });
const learningEmitter = new EventEmitter();
let tfidf = new TfIdf();
let patternsCache = [];

async function loadPatternsAndTrainModels(maxDocs = 300) {
    await client.connect();
    const db = client.db('CASIDB');
    const patterns = db.collection('patterns');
    const allPatterns = await patterns.find({}).limit(maxDocs).toArray();

    nlpManager = new NlpManager({ languages: ['en'], forceNER: true });
    tfidf = new TfIdf();
    patternsCache = [];

    let count = 0;
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
            tfidf.addDocument(text);
            patternsCache.push({ concept: pattern.concept, text, pattern });
            count++;
            if (count >= maxDocs) break;
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
        tfidf.addDocument(text);
        patternsCache.push({ concept: pattern.concept, text, pattern });
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

function distillOkeyAIResponse(okeyText, patterns, prompt) {
    tfidf.addDocument(okeyText);
    let keywords = [];
    tfidf.listTerms(tfidf.documents.length - 1).slice(0, 10).forEach(item => {
        keywords.push(item.term);
    });
    tfidf.documents.pop();

    let bestMatch = null;
    let bestScore = 0;
    for (const p of patterns) {
        let score = 0;
        for (const kw of keywords) {
            if (p.text.toLowerCase().includes(kw)) score++;
        }
        if (score > bestScore) {
            bestScore = score;
            bestMatch = p;
        }
    }

    let base = prompt;
    if (bestMatch && bestMatch.text) {
        base = bestMatch.text;
    }
    let summary = '';
    if (keywords.length > 0) {
        summary = keywords.slice(0, 5).map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(', ') + '. ';
    }
    let doc = nlp(base);
    let sentences = doc.sentences().out('array');
    let chosen = sentences.length > 0 ? sentences[0] : base;
    let result = summary + chosen;
    result = result.replace(/\s{2,}/g, ' ').trim();
    if (!/[.!?]$/.test(result)) result += '.';
    return result;
}

async function fetchOkeyAIResponse(prompt) {
    const apiUrl = 'https://api.okeymeta.com.ng/api/ssailm/model/okeyai3.0-vanguard/okeyai';
    const params = { input: prompt };
    try {
        const response = await axios.get(apiUrl, { params, timeout: 60000 });
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
    } catch {
        return '';
    }
}

async function generateResponse({ prompt, diversityFactor = 0.5, depth = 10, breadth = 5, maxWords = 100, mood = 'neutral', patterns }) {
    try {
        if (!prompt || typeof prompt !== 'string' || prompt.trim().length < 1) {
            throw new Error('Prompt must be a non-empty string');
        }

        if (!nlpManager || !tfidf || patternsCache.length === 0) {
            await loadPatternsAndTrainModels();
        }

        const okeyText = await fetchOkeyAIResponse(prompt);
        let distilled = '';
        if (okeyText) {
            distilled = distillOkeyAIResponse(okeyText, patternsCache, prompt);
            await client.connect();
            const db = client.db('CASIDB');
            const patternsCol = db.collection('patterns');
            await patternsCol.insertOne({
                concept: prompt,
                content: zlib.gzipSync(okeyText).toString('base64'),
                entities: [],
                sentiment: 0,
                updatedAt: new Date(),
                confidence: 0.97,
                source: 'OkeyMetaAI',
                title: `OkeyMetaAI: ${prompt}`
            });
            learningEmitter.emit('newPattern', {
                concept: prompt,
                content: zlib.gzipSync(okeyText).toString('base64')
            });
            console.log('[CASI] Learned from OkeyAI:', okeyText);
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

        let outputText = distilled || fallbackText || "I'm not sure, but let's explore this together!";
        outputText = formatResponse(outputText, maxWords, mood);

        const confidence = okeyText ? 0.98 : 0.9;
        const outputId = crypto.randomBytes(12).toString('hex');

        await client.connect();
        const db = client.db('CASIDB');
        const patternsCol = db.collection('patterns');
        await patternsCol.insertOne({
            concept: prompt,
            content: zlib.gzipSync(outputText).toString('base64'),
            entities: [],
            sentiment: 0,
            updatedAt: new Date(),
            confidence,
            source: 'CASI',
            title: `Response to: ${prompt.slice(0, 50)}`
        });

        learningEmitter.emit('newPattern', {
            concept: prompt,
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