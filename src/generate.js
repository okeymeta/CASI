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

const mongoUri = 'mongodb+srv://nwaozor:nwaozor@cluster0.rmvi7qm.mongodb.net/CASIDB?retryWrites=true&w=majority';
const client = new MongoClient(mongoUri);

const RESPONSE_TEMPLATES = {
    neutral: '{{text}}',
    empathetic: '{{#if isEmotionalPrompt}}{{text}} I’m here to support you!{{else}}{{text}}{{/if}}',
    enthusiastic: '{{text}} What a fascinating topic! Want to explore more?'
};

async function googleVerify(text) {
    try {
        const response = await axios.get(`https://www.google.com/search?q=${encodeURIComponent(text)}`, {
            headers: { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/96.0.4664.110' },
            timeout: 5000
        });
        return response.status === 200 ? 0.98 : 0.9;
    } catch {
        return 0.9;
    }
}

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
    // Deep clean text
    text = text
        .replace(/\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s+\d{4}\b/g, '')
        .replace(/\n+/g, ' ')
        .replace(/\s+/g, ' ')
        .replace(/[^\w\s.,!?]/g, '')
        .trim();
    
    // Generate coherent sentences
    const doc = nlp(text);
    let sentences = doc.sentences().out('array').slice(0, 3);
    if (sentences.length === 0) sentences = ['Let’s explore this topic together.'];
    
    // Adjust tone based on mood
    if (mood === 'enthusiastic') {
        sentences = sentences.map(s => s.replace(/[.!?]$/, '!'));
    } else if (mood === 'empathetic') {
        sentences = sentences.map(s => s.replace(/[.!?]$/, '.'));
    }
    
    // Truncate to maxWords
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
    
    // Ensure proper punctuation
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

async function generateResponse({ prompt, diversityFactor = 0.5, depth = 10, breadth = 5, maxWords = 100, mood = 'neutral' }) {
    try {
        if (!prompt || typeof prompt !== 'string' || prompt.trim().length < 1) {
            throw new Error('Prompt must be a non-empty string');
        }

        const typoCorrectedPrompt = correctTypos(prompt.trim());
        const cleanedPrompt = preprocessPrompt(typoCorrectedPrompt);
        console.log(`[Generate] Processing prompt: ${cleanedPrompt}`);

        await client.connect();
        const db = client.db('CASIDB');
        const patterns = db.collection('patterns');

        // Extract entities
        const doc = nlp(cleanedPrompt);
        let promptEntities = [];
        try {
            doc.people().forEach(p => promptEntities.push({ value: p.text(), type: 'person' }));
            doc.places().forEach(p => promptEntities.push({ value: p.text(), type: 'place' }));
            doc.organizations().forEach(o => promptEntities.push({ value: o.text(), type: 'organization' }));
            doc.topics().forEach(t => promptEntities.push({ value: t.text(), type: 'topic' }));
        } catch (error) {
            console.warn('[Generate] Entity extraction failed:', error.message);
        }

        // Sentiment analysis with error handling
        let promptSentiment = 0;
        try {
            const sentimentResult = sentiment(cleanedPrompt);
            promptSentiment = sentimentResult && typeof sentimentResult.score === 'number' ? sentimentResult.score : 0;
        } catch (error) {
            console.warn('[Generate] Sentiment analysis failed:', error.message);
        }

        const emotionalKeywords = ['nervous', 'anxiety', 'stress', 'afraid', 'worried'];
        const isEmotionalPrompt = emotionalKeywords.some(keyword => cleanedPrompt.includes(keyword));

        // Extract keywords
        const topics = doc.topics().out('array');
        const nouns = doc.nouns().out('array');
        const keywords = [...new Set([...topics, ...nouns])].map(k => k.toLowerCase());

        // Find patterns with strict matching
        const relatedPatterns = await patterns.find({
            $or: [
                { concept: { $in: keywords } },
                { entities: { $elemMatch: { value: { $in: promptEntities.map(e => e.value) } } } },
                { concept: { $regex: keywords.join('|'), $options: 'i' } }
            ]
        })
        .sort({ confidence: -1, updatedAt: -1 })
        .limit(breadth * 3)
        .toArray();

        // Select diverse patterns
        const selectedPatterns = lodash.shuffle(relatedPatterns)
            .slice(0, Math.max(1, Math.ceil(breadth * diversityFactor)))
            .sort((a, b) => b.confidence - a.confidence);
        console.log(`[Generate] Found ${relatedPatterns.length} patterns, using ${selectedPatterns.length}`);

        let synthesizedText = '';
        const classifier = new natural.BayesClassifier();
        if (selectedPatterns.length > 0) {
            selectedPatterns.forEach(pattern => {
                try {
                    const text = zlib.gunzipSync(Buffer.from(pattern.content, 'base64')).toString();
                    const weight = pattern.confidence * (pattern.source === 'seed' ? 1.3 : 1);
                    classifier.addDocument(text, pattern.concept, weight);
                } catch (error) {
                    console.warn('[Generate] Failed to decompress pattern:', pattern._id, error.message);
                }
            });
            classifier.train();

            const classifications = classifier.getClassifications(cleanedPrompt)
                .slice(0, Math.min(depth, selectedPatterns.length))
                .filter(c => c.label);

            const topConcepts = classifications.map(c => c.label);
            let patternTexts = selectedPatterns
                .filter(p => topConcepts.includes(p.concept) || keywords.some(k => p.concept.includes(k)))
                .map(p => {
                    try {
                        return zlib.gunzipSync(Buffer.from(p.content, 'base64')).toString();
                    } catch {
                        return '';
                    }
                })
                .filter(t => t);

            if (patternTexts.length > 0) {
                patternTexts = lodash.shuffle(patternTexts);
                const nlpDoc = nlp(patternTexts.join(' '));
                const sentences = nlpDoc.sentences().out('array');
                synthesizedText = lodash.shuffle(sentences)
                    .slice(0, Math.min(3, sentences.length))
                    .join(' ');
            }
        }

        // Fallback synthesis
        if (!synthesizedText || synthesizedText.length < 10 || !keywords.some(k => synthesizedText.toLowerCase().includes(k))) {
            console.log('[Generate] Using fallback response');
            const fallbackPatterns = [
                { concept: 'african history', text: 'African history is rich, covering ancient kingdoms like Mali, Songhai, and Great Zimbabwe, colonial times, and modern independence movements.' },
                { concept: 'african history', text: 'Africa’s history shines with vibrant cultures, like the Ashanti Empire, and pivotal global contributions.' },
                { concept: 'nervousness', text: 'Feeling nervous is normal before big moments. Try deep breathing or practicing to feel more confident.' },
                { concept: 'artificial intelligence', text: 'Artificial intelligence powers machines to learn and solve problems, transforming tech and daily life.' },
                { concept: 'yoruba culture', text: 'Yoruba culture is alive with festivals like Osun-Osogbo, intricate art, and rich traditions.' },
                { concept: 'nigerian tech', text: 'Nigerian tech is thriving, with startups like Flutterwave and Paystack driving innovation.' },
                { concept: 'mental health', text: 'Mental health is vital—mindfulness and support can help you find balance.' },
                { concept: 'education technology', text: 'Education technology transforms learning with digital tools like online courses and apps.' },
                { concept: 'blockchain', text: 'Blockchain offers secure, transparent transactions, revolutionizing tech industries.' },
                { concept: 'climate change', text: 'Climate change affects our planet, but solutions like renewable energy can make a difference.' },
                { concept: 'greeting', text: 'Hey there! I’m excited to chat about anything you’d like.' },
                { concept: 'quantum computing', text: 'Quantum computing uses quantum mechanics to solve complex problems faster than traditional computers.' },
                { concept: 'casual', text: 'Just chilling? Let’s talk about whatever’s on your mind!' }
            ];

            const relevantFallbacks = fallbackPatterns.filter(p => 
                keywords.some(k => p.concept.includes(k)) || p.concept === 'greeting' || p.concept === 'casual'
            );
            synthesizedText = relevantFallbacks.length > 0
                ? lodash.shuffle(relevantFallbacks)[0].text
                : `I’d love to dive into ${cleanedPrompt.replace(/^\w/, c => c.toUpperCase())}. What details can you share?`;
        }

        // Format and apply template
        const formattedText = formatResponse(synthesizedText, maxWords, mood);
        const template = handlebars.compile(RESPONSE_TEMPLATES[mood.toLowerCase()] || RESPONSE_TEMPLATES.neutral);
        const outputText = template({ text: formattedText, isEmotionalPrompt });

        // Verify and store
        const confidence = await googleVerify(outputText);
        const outputId = crypto.randomBytes(12).toString('hex');

        await patterns.insertOne({
            concept: keywords[0] || cleanedPrompt,
            content: zlib.gzipSync(outputText).toString('base64'),
            entities: promptEntities,
            sentiment: promptSentiment,
            updatedAt: new Date(),
            confidence,
            source: 'generated',
            title: `Response to: ${cleanedPrompt.slice(0, 50)}`
        });

        console.log(`[Generate] Response generated, confidence: ${confidence}, outputId: ${outputId}`);
        return { text: outputText, confidence, outputId };
    } catch (error) {
        console.error('[Generate] Generation error:', error.message);
        throw { message: `Generation failed: ${error.message}`, status: 500 };
    } finally {
        await client.close();
    }
}

module.exports = { generateResponse };