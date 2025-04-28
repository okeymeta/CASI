const axios = require('axios');
const nlp = require('compromise');
const sentiment = require('sentiment');
const { MongoClient } = require('mongodb');
const zlib = require('zlib');
const winston = require('winston');
const TfIdf = require('node-tfidf');
const { paraphraseSentence } = require('./generate');

const mongoUri = 'mongodb+srv://nwaozor:nwaozor@cluster0.rmvi7qm.mongodb.net/CASIDB?retryWrites=true&w=majority';
const client = new MongoClient(mongoUri);

// Logger setup
const logger = winston.createLogger({
    transports: [
        new winston.transports.File({ filename: 'error.log' }),
        new winston.transports.File({ filename: 'combined.log' }),
        new winston.transports.Console()
    ]
});

const RATE_LIMIT_MS = 3000;
const MAX_RETRIES = 3;
const MAX_RESULTS_PER_PROMPT = 1;

// Expanded prompts for conversational and diverse training
const PROMPTS = [
    'Hello',
    'Hi, how are you?',
    'Hey, what can you do?',
    'Greetings, tell me about yourself.',
    'Good morning, how’s it going?',
    'What’s up?',
    'How to start a conversation?',
    'What is a good way to say hello?',
    'How to respond to greetings?',
    'Explain artificial intelligence in simple terms.',
    'What is the importance of empathy in communication?',
    'How does blockchain technology work?',
    'Describe the benefits of mindfulness.',
    'Tell me a joke about computers.',
    'How can I improve my productivity?',
    'What are the main causes of climate change?',
    'Summarize the history of the Yoruba people.',
    'How do neural networks learn?',
    'Give me tips for public speaking.',
    'What is quantum computing?',
    'How to manage stress effectively?',
    'What is the difference between machine learning and deep learning?',
    'Explain the concept of cultural heritage.',
    'How to make friends as an adult?',
    'Describe the process of critical thinking.',
    'What are the advantages of renewable energy?',
    'How to solve a conflict peacefully?',
    'What is the role of nutrition in health?',
    'Explain the basics of coding to a beginner.',
    'How does sleep affect mental health?',
    'What is the significance of African literature?',
    'How to motivate yourself after failure?',
    'What is the future of artificial intelligence?',
    'How to apologize sincerely?',
    'Describe the impact of social media on society.',
    'What are the key elements of storytelling?',
    'How to be creative in problem solving?',
    'Explain the ethics of AI.',
    'What is the importance of self-awareness?',
    'How to balance work and life?',
    'What is the meaning of friendship?',
    'How to explain blockchain to a child?',
    'What are the benefits of exercise?',
    'How to learn new skills quickly?',
    'What is the difference between knowledge and wisdom?',
    'How to help someone in need?',
    'Explain the basics of robotics.',
    'What is the impact of AI on jobs?',
    'How to plan and achieve goals?',
    'Describe the process of decision making.'
];

// Utility: Remove personal emails and phone numbers
function filterSensitiveInfo(text) {
    text = text.replace(/(\+?\d{1,3}[-.\s]?)?(\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{4,}/g, '[filtered]');
    text = text.replace(/\b[A-Za-z0-9._%+-]+@(gmail|yahoo|outlook|hotmail|protonmail|icloud|aol|mail|zoho|gmx|yandex|qq|163|126|sina|yeah|foxmail|googlemail)\.[A-Za-z]{2,}\b/gi, '[filtered]');
    return text;
}

// Utility: Clean up text
function cleanAIText(text) {
    text = text.replace(/ |nbsp;/gi, ' ');
    text = text.replace(/&[a-z]+;/gi, ' ');
    text = text.replace(/([a-z])([A-Z])/g, '$1 $2');
    text = text.replace(/\b(\w+)(?:\s+\1\b)+/gi, '$1');
    text = text.replace(/\s{2,}/g, ' ');
    text = text.replace(/[,;:\-]+$/, '');
    text = text.trim();
    if (text && !/[.!?]$/.test(text)) text += '.';
    text = text.charAt(0).toUpperCase() + text.slice(1);
    return text;
}

async function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function dropIndexIfExists(collection, indexName) {
    try {
        await collection.dropIndex(indexName);
        logger.info(`Dropped ${indexName} index`);
    } catch (error) {
        if (error.codeName === 'IndexNotFound') {
            logger.info(`${indexName} index not found, proceeding`);
        } else {
            logger.warn(`Failed to drop ${indexName} index: ${error.message}`);
        }
    }
}

async function learnFromOkeyMetaAI(prompt, patterns) {
    logger.info(`[CASI] Learning from OkeyMetaAI for prompt: "${prompt}"`);
    try {
        const apiUrl = 'https://api.okeymeta.com.ng/api/ssailm/model/okeyai3.0-vanguard/okeyai';
        const params = { input: prompt };
        const response = await axios.get(apiUrl, { params, timeout: 60000 });

        let aiData = response.data;
        if (typeof aiData === 'string') {
            try {
                aiData = JSON.parse(aiData);
            } catch {
                logger.warn(`OkeyMetaAI returned non-JSON string for prompt: ${prompt}. Raw: ${aiData}`);
                return [];
            }
        }

        const contentRaw = aiData && (aiData.response || aiData?.model?.response);
        if (!contentRaw || typeof contentRaw !== 'string' || contentRaw.trim().length < 5) {
            logger.warn(`OkeyMetaAI returned no usable response for prompt: ${prompt}. Raw response: ${JSON.stringify(response.data)}`);
            return [];
        }

        let content = contentRaw.trim();
        content = filterSensitiveInfo(content);
        content = cleanAIText(content);

        // Distill using TF-IDF and NLP
        const tfidf = new TfIdf();
        tfidf.addDocument(content);
        let keywords = [];
        tfidf.listTerms(0).slice(0, 3).forEach(item => keywords.push(item.term));
        const nlpDoc = nlp(content);
        const sentences = nlpDoc.sentences().out('array');
        let distilled = '';
        if (keywords.length > 0 && sentences[0]) {
            distilled = `${keywords[0].charAt(0).toUpperCase() + keywords[0].slice(1)}: ${sentences[0]}`;
        } else {
            distilled = sentences[0] || content;
        }
        distilled = cleanAIText(distilled);

        // Generate CASI's unique version
        let casiOwn = distilled;
        if (sentences.length > 1 && !['hello', 'hi', 'hey', 'greetings'].includes(prompt.toLowerCase())) {
            try {
                casiOwn = await paraphraseSentence(sentences[0]);
            } catch (error) {
                logger.warn(`Paraphrasing failed for "${sentences[0]}": ${error.message}`);
                casiOwn = sentences[0]; // Fallback to original
            }
        } else if (['hello', 'hi', 'hey', 'greetings'].includes(prompt.toLowerCase())) {
            casiOwn = 'Hi! I’m ready to assist you.';
        }
        casiOwn = cleanAIText(casiOwn);

        const compressedContent = zlib.gzipSync(content).toString('base64');
        const compressedCasiOwn = zlib.gzipSync(casiOwn).toString('base64');
        const entities = nlpDoc.topics().out('array');
        let sentimentScore = 0;
        try {
            const sentimentResult = sentiment(content);
            sentimentScore = sentimentResult && typeof sentimentResult.score === 'number' ? sentimentResult.score : 0;
        } catch {}

        // Store only if distinct and meaningful
        if (casiOwn !== content && casiOwn.length > 5) {
            await patterns.insertOne({
                url: `${apiUrl}?input=${encodeURIComponent(prompt)}`,
                prompt,
                entities,
                sentiment: sentimentScore,
                content: compressedContent,
                distilled: compressedCasiOwn,
                concept: prompt,
                source: 'OkeyMetaAI',
                title: `OkeyMetaAI: ${prompt}`,
                createdAt: new Date(),
                updatedAt: new Date(),
                confidence: Math.min(0.97 + sentimentScore / 100, 0.99)
            });
            await patterns.insertOne({
                url: `${apiUrl}?input=${encodeURIComponent(prompt)}`,
                prompt,
                entities,
                sentiment: sentimentScore,
                content: compressedCasiOwn,
                concept: prompt,
                source: 'CASI',
                title: `CASI: ${prompt}`,
                createdAt: new Date(),
                updatedAt: new Date(),
                confidence: Math.min(0.98 + sentimentScore / 100, 0.995)
            });

            logger.info(`[CASI] Learned and distilled: ${casiOwn}`);
            return [{
                status: 'learned',
                nodesAdded: 2,
                confidence: Math.min(0.98 + sentimentScore / 100, 0.995),
                source: 'CASI',
                title: `CASI: ${prompt}`,
                url: `${apiUrl}?input=${encodeURIComponent(prompt)}`,
                distilled: casiOwn
            }];
        } else {
            logger.warn(`[CASI] Skipped storing identical or short distilled response for prompt: ${prompt}`);
            return [];
        }
    } catch (error) {
        logger.error(`[CASI] OkeyMetaAI error for prompt: ${prompt}: ${error.message}`);
        return [];
    }
}

async function scrapeOnDemand(query, prompt, patterns) {
    try {
        // Skip scraping for paraphrasing prompts
        if (prompt.toLowerCase().startsWith('paraphrase:')) {
            logger.info(`[CASI] Skipping on-demand scrape for paraphrasing prompt: ${prompt}`);
            return [];
        }
        logger.info(`[CASI] On-demand learning for prompt: ${prompt}`);
        const results = await learnFromOkeyMetaAI(prompt, patterns);
        logger.info(`[CASI] On-demand learning completed for "${prompt}", added ${results.length} results`);
        return results;
    } catch (error) {
        logger.error(`[CASI] On-demand learning failed for "${prompt}": ${error.message}`);
        return [];
    }
}

async function scrapeAll(prompt = 'Learn about diverse topics including AI, culture, tech, health, and more') {
    try {
        await client.connect();
        const db = client.db('CASIDB');
        const patterns = db.collection('patterns');
        await dropIndexIfExists(patterns, 'concept_1');

        await patterns.createIndex({ concept: 1, updatedAt: -1 });
        await patterns.createIndex({ source: 1 });
        await patterns.createIndex({ entities: 1 });

        const results = [];
        for (const p of PROMPTS) {
            logger.info(`[CASI] Learning for prompt: ${p}`);
            const aiResults = await learnFromOkeyMetaAI(p, patterns);
            results.push(...aiResults);
            await delay(RATE_LIMIT_MS);
        }

        logger.info(`[CASI] Learning completed, added ${results.length} results`);
        return results;
    } catch (error) {
        logger.error(`[CASI] Learning error: ${error.message}`);
        throw error;
    } finally {
        await client.close();
    }
}

module.exports = { scrapeAll, scrapeOnDemand };