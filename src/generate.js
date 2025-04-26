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
        .replace(/[^\w\s.,!?*'-]/g, '') // Preserve list and bold markers
        .replace(/\.\.+/g, '.')
        .trim()
        .toLowerCase();
}

function preserveFormatting(text) {
    // Preserve list markers and bold formatting
    text = text
        .replace(/(\n\s*[-*]\s*)/g, '\n$1') // Keep bullet points
        .replace(/(\n\s*\d+\.\s*)/g, '\n$1') // Keep numbered lists
        .replace(/(\*\*[^\*]+\*\*)/g, '$1') // Keep bold markdown
        .replace(/(\*[^\*]+\*)/g, '$1'); // Keep italic markdown
    
    // Normalize line breaks to single newlines
    text = text.replace(/\n\s*\n+/g, '\n');
    
    // Remove extra spaces around punctuation
    text = text.replace(/\s+([.,!?])/g, '$1');
    
    return text.trim();
}

function formatResponse(text, maxWords, mood) {
    // Preserve formatting from OkeyAI
    text = preserveFormatting(text);
    
    // Basic cleaning
    text = text
        .replace(/\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s+\d{4}\b/g, '')
        .replace(/[^\w\s.,!?*'\n-]/g, '') // Preserve newlines and markdown
        .trim();
    
    const doc = nlp(text);
    let sentences = doc.sentences().out('array');
    if (sentences.length === 0) sentences = ['Let’s explore this topic together.'];
    
    // Split text into lines to handle lists
    const lines = text.split('\n').filter(line => line.trim());
    let isList = lines.some(line => /^\s*(\d+\.\s+|[-*]\s+)/.test(line));
    
    // Apply mood adjustments
    if (!isList) {
        if (mood === 'enthusiastic') {
            sentences = sentences.map(s => s.replace(/[.!?]$/, '!'));
        } else if (mood === 'empathetic') {
            sentences = sentences.map(s => s.replace(/[.!?]$/, '.'));
        }
    }
    
    // Handle lists explicitly
    let outputLines = [];
    let wordCount = 0;
    if (isList) {
        for (const line of lines) {
            const words = line.split(' ');
            if (wordCount + words.length <= maxWords) {
                outputLines.push(line);
                wordCount += words.length;
            } else {
                break;
            }
        }
    } else {
        for (const sentence of sentences) {
            const words = sentence.split(' ');
            if (wordCount + words.length <= maxWords) {
                outputLines.push(sentence);
                wordCount += words.length;
            } else {
                break;
            }
        }
    }
    
    let finalText = outputLines.join(isList ? '\n' : ' ');
    if (!finalText) finalText = 'Let’s dive into this topic!';
    
    // Capitalize first letter and ensure ending punctuation
    if (!isList) {
        finalText = finalText.charAt(0).toUpperCase() + finalText.slice(1);
        if (!/[.!?]/.test(finalText.slice(-1))) finalText += '.';
    }
    
    // Ensure grammatical correctness
    const finalDoc = nlp(finalText);
    finalText = finalDoc.text();
    
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

function synthesizeFromOkeyAI(okeyText, prompt, patterns) {
    // Preserve original formatting
    okeyText = preserveFormatting(okeyText);
    
    const doc = nlp(okeyText);
    const sentences = doc.sentences().out('array');
    const entities = doc.topics().out('array').concat(doc.people().out('array'), doc.places().out('array'));
    const promptDoc = nlp(prompt);
    const promptEntities = promptDoc.topics().out('array').concat(promptDoc.people().out('array'), promptDoc.places().out('array'));

    // Extract keywords with higher specificity
    tfidf.addDocument(okeyText);
    let keywords = [];
    tfidf.listTerms(tfidf.documents.length - 1).slice(0, 15).forEach(item => {
        if (item.tfidf > 1.5) keywords.push(item.term);
    });
    tfidf.documents.pop();

    // Find best matching pattern
    let bestMatch = null;
    let bestScore = 0;
    for (const p of patterns) {
        let score = 0;
        for (const kw of keywords) {
            if (p.text.toLowerCase().includes(kw)) score += 2;
        }
        for (const entity of entities) {
            if (p.text.toLowerCase().includes(entity.toLowerCase())) score += 1;
        }
        if (score > bestScore) {
            bestScore = score;
            bestMatch = p;
        }
    }

    // Build response
    let base = okeyText;
    if (bestMatch && bestMatch.text && bestScore > 0) {
        const matchSentences = nlp(bestMatch.text).sentences().out('array');
        // Combine intelligently, preserving list structure
        const isOkeyList = okeyText.split('\n').some(line => /^\s*(\d+\.\s+|[-*]\s+)/.test(line));
        if (isOkeyList) {
            base = okeyText + '\n' + matchSentences.join(' ');
        } else {
            base = sentences.concat(matchSentences).slice(0, 5).join(' ');
        }
    }

    let result = base;
    // Ensure prompt entities are included if missing
    if (promptEntities.length > 0 && !result.toLowerCase().includes(promptEntities[0].toLowerCase())) {
        result = `${promptEntities[0]}: ${result}`;
    }

    // Ensure minimum length and coherence
    if (result.split(' ').length < 20 && sentences.length > 2) {
        result = sentences.slice(0, 3).join(' ');
    }

    // Final cleaning
    result = result.replace(/\b(\w+)(?:\s+\1\b)+/gi, '$1'); // Remove duplicate words
    result = result.replace(/\s{2,}/g, ' ').trim();
    if (!result.split('\n').some(line => /^\s*(\d+\.\s+|[-*]\s+)/.test(line))) {
        result = result.charAt(0).toUpperCase() + result.slice(1);
        if (!/[.!?]$/.test(result)) result += '.';
    }

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

        const cleanedPrompt = preprocessPrompt(prompt);
        const okeyText = await fetchOkeyAIResponse(cleanedPrompt);
        let synthesized = '';
        if (okeyText) {
            synthesized = synthesizeFromOkeyAI(okeyText, cleanedPrompt, patternsCache);
            await client.connect();
            const db = client.db('CASIDB');
            const patternsCol = db.collection('patterns');
            await patternsCol.insertOne({
                concept: cleanedPrompt,
                content: zlib.gzipSync(okeyText).toString('base64'),
                entities: [],
                sentiment: 0,
                updatedAt: new Date(),
                confidence: 0.97,
                source: 'OkeyMetaAI',
                title: `OkeyMetaAI: ${cleanedPrompt.slice(0, 50)}`
            });
            learningEmitter.emit('newPattern', {
                concept: cleanedPrompt,
                content: zlib.gzipSync(okeyText).toString('base64')
            });
            console.log('[CASI] Learned from OkeyAI:', okeyText);
        }

        const nlpResult = await nlpManager.process('en', cleanedPrompt);

        let fallbackText = '';
        if (patterns && patterns.length > 0) {
            const best = patterns.find(p =>
                (nlpResult.intent && p.concept === nlpResult.intent) ||
                (p.concept && cleanedPrompt.toLowerCase().includes(p.concept.toLowerCase()))
            );
            if (best) {
                try {
                    fallbackText = zlib.gunzipSync(Buffer.from(best.content, 'base64')).toString();
                } catch {
                    fallbackText = '';
                }
            }
        }

        let outputText = synthesized || fallbackText || "I'm not sure, but let's explore this together!";
        outputText = formatResponse(outputText, maxWords, mood);

        const confidence = okeyText ? 0.98 : 0.9;
        const outputId = crypto.randomBytes(12).toString('hex');

        await client.connect();
        const db = client.db('CASIDB');
        const patternsCol = db.collection('patterns');
        await patternsCol.insertOne({
            concept: cleanedPrompt,
            content: zlib.gzipSync(outputText).toString('base64'),
            entities: [],
            sentiment: 0,
            updatedAt: new Date(),
            confidence,
            source: 'CASI',
            title: `Response to: ${cleanedPrompt.slice(0, 50)}`
        });

        learningEmitter.emit('newPattern', {
            concept: cleanedPrompt,
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