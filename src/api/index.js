const http = require('http');
const { parse } = require('url');
const { MongoClient } = require('mongodb');
const { validatePrompt, validateGenerationParams, validateUrl, validateFeedback } = require('../utils/validate');
const WebScraper = require('../scrape');
const Generator = require('../generate');
const lattice = require('../lattice');

const mongoUri = 'mongodb+srv://nwaozor:nwaozor@cluster0.rmvi7qm.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0';
const client = new MongoClient(mongoUri, { maxPoolSize: 10, retryWrites: true });

const responseCache = new Map();
const cacheResponse = (key, data, duration) => {
    responseCache.set(key, { data, timestamp: Date.now() + duration });
};
const getCachedResponse = (key) => {
    const cached = responseCache.get(key);
    if (cached && Date.now() < cached.timestamp) {
        return cached.data;
    }
    return null;
};

const server = http.createServer(async (req, res) => {
    const { pathname } = parse(req.url, true);
    let body = '';

    req.on('data', chunk => body += chunk);
    req.on('end', async () => {
        try {
            res.setHeader('Content-Type', 'application/json');
            let response;

            if (req.method === 'POST' && pathname === '/api/scrape') {
                const data = JSON.parse(body || '{}');
                const url = validateUrl(data.url);
                const prompt = validatePrompt(data.prompt);
                response = await WebScraper.scrapeAndLearn(url, prompt);
            } else if (req.method === 'POST' && pathname === '/api/scrape-all') {
                const data = JSON.parse(body || '{}');
                const prompt = validatePrompt(data.prompt || 'Learn about diverse topics');
                response = await WebScraper.scrapeAll(prompt);
            } else if (req.method === 'POST' && pathname === '/api/generate') {
                const data = JSON.parse(body || '{}');
                const prompt = validatePrompt(data.prompt);
                const params = validateGenerationParams(data);
                const cacheKey = `${pathname}:${prompt}:${JSON.stringify(params)}`;
                response = getCachedResponse(cacheKey);
                if (!response) {
                    response = await Generator.generateResponse({ prompt, ...params });
                    cacheResponse(cacheKey, response, 3600 * 1000);
                }
            } else if (req.method === 'POST' && pathname === '/api/feedback') {
                const data = JSON.parse(body || '{}');
                const feedback = validateFeedback(data);
                await client.connect();
                const db = client.db('CASIDB');
                const result = await db.collection('feedback').insertOne({
                    ...feedback,
                    timestamp: new Date(),
                    processed: false
                });
                await lattice.updateNode({
                    concept: feedback.outputId,
                    updates: {
                        confidence: feedback.isAccurate ? 0.98 : 0.9,
                        verified: feedback.isAccurate
                    }
                });
                response = { status: 'refined', updatedNodes: result.insertedCount };
            } else if (req.method === 'GET' && pathname === '/api/whoami') {
                response = { text: "I’m CASI, a Synthesis-Driven Intelligence by xAI Community, revolutionizing AI with 98% accuracy." };
            } else if (req.method === 'POST' && pathname === '/api/solve') {
                const data = JSON.parse(body || '{}');
                const problem = validatePrompt(data.problem);
                const context = validatePrompt(data.context || '');
                response = await Generator.generateResponse({
                    prompt: `${problem} ${context}`,
                    mood: 'neutral',
                    maxWords: 500
                });
            } else if (req.method === 'GET' && pathname === '/api/rules') {
                response = { text: "CASI ensures no PII, rate-limited scraping, ToS compliance." };
            } else if (req.method === 'GET' && pathname === '/api/health') {
                response = { status: 'healthy', version: '1.0.0', timestamp: new Date() };
            } else {
                response = { error: 'Not found', status: 404, timestamp: new Date() };
                res.statusCode = 404;
            }

            res.end(JSON.stringify(response));
        } catch (error) {
            console.error('Server error:', error);
            res.statusCode = error.status || 500;
            res.end(JSON.stringify({ error: error.message, status: error.status || 500, timestamp: new Date() }));
        }
    });
});

const init = async () => {
    try {
        console.log('Connecting to MongoDB...');
        await client.connect();
        console.log('MongoDB connected');
        if (typeof lattice.init !== 'function') {
            throw new Error('lattice.init is not a function. Check src/lattice.js');
        }
        await lattice.init();
        console.log('Lattice initialized');
        await WebScraper.scrapeAll('Learn about diverse topics');
        console.log('Initial scraping completed');
        server.listen(3000, () => {
            console.log('CASI server started on http://localhost:3000');
        });
    } catch (error) {
        console.error('Initialization error:', error);
        process.exit(1);
    }
};

init();