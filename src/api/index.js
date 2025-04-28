const express = require('express');
const { MongoClient } = require('mongodb');
const { scrapeAll, scrapeOnDemand } = require('../scrape');
const { generateResponse, loadPatternsAndTrainModels, initializeModels } = require('../generate');
const { init: initLattice } = require('../lattice');
const winston = require('winston');
const crypto = require('crypto');

const app = express();
const port = process.env.PORT || 3000;
const mongoUri = 'mongodb+srv://nwaozor:nwaozor@cluster0.rmvi7qm.mongodb.net/CASIDB?retryWrites=true&w=majority';
const client = new MongoClient(mongoUri);

const logger = winston.createLogger({
    transports: [
        new winston.transports.File({ filename: 'error.log' }),
        new winston.transports.File({ filename: 'combined.log' }),
        new winston.transports.Console()
    ]
});

app.use(express.json());

async function init() {
    try {
        console.log('Connecting to MongoDB...');
        await client.connect();
        console.log('MongoDB connected');

        try {
            const { patterns } = await initLattice();
            app.locals.patterns = patterns;
        } catch (error) {
            logger.warn('Lattice initialization failed, using MongoDB patterns:', error.message);
            const db = client.db('CASIDB');
            app.locals.patterns = db.collection('patterns');
        }

        try {
            await initializeModels();
            console.log('ML models loaded');
        } catch (error) {
            logger.error('Failed to load ML models:', error.message);
            console.warn('Proceeding with limited functionality');
        }

        try {
            await loadPatternsAndTrainModels();
            console.log('Patterns and models loaded');
        } catch (error) {
            logger.error('Failed to load patterns and models:', error.message);
            console.warn('Proceeding with basic NLP functionality');
        }

        setInterval(async () => {
            try {
                const results = await scrapeAll('Learn about diverse topics including AI, culture, tech, health, and more');
                logger.info(`Periodic scraping completed, added ${results.length} results`);
                await loadPatternsAndTrainModels();
            } catch (error) {
                logger.error('Periodic scraping failed:', error.message);
            }
        }, 6 * 60 * 60 * 1000);

        const results = await scrapeAll('Learn about diverse topics including AI, culture, tech, health, and more');
        logger.info(`Initial scraping completed, added ${results.length} results`);
        await loadPatternsAndTrainModels();
    } catch (error) {
        logger.error('Initialization error:', error);
        throw error;
    }
}

app.post('/api/scrape-all', async (req, res) => {
    try {
        const { prompt } = req.body;
        if (!prompt) {
            return res.status(400).json({ error: 'Prompt is required' });
        }
        const results = await scrapeAll(prompt);
        logger.info(`Manual scrape-all completed, added ${results.length} results`);
        await loadPatternsAndTrainModels();
        res.json(results);
    } catch (error) {
        logger.error(`Scrape-all error: ${error.message}`);
        res.status(500).json({ error: error.message });
    }
});

app.post('/api/generate', async (req, res) => {
    try {
        const { 
            prompt, 
            diversityFactor = 0.7, 
            depth = 8, 
            breadth = 6, 
            maxWords = 300, 
            mood = 'neutral',
            top_k = 50,
            temperature = 0.7,
            max_tokens = 100
        } = req.body;

        if (!prompt || typeof prompt !== 'string' || prompt.trim().length < 1) {
            return res.status(400).json({ error: 'Prompt is required' });
        }

        // Validate new parameters
        if (typeof top_k !== 'number' || top_k < 1 || top_k > 100) {
            return res.status(400).json({ error: 'top_k must be a number between 1 and 100' });
        }
        if (typeof temperature !== 'number' || temperature < 0.1 || temperature > 2.0) {
            return res.status(400).json({ error: 'temperature must be a number between 0.1 and 2.0' });
        }
        if (typeof max_tokens !== 'number' || max_tokens < 10 || max_tokens > 512) {
            return res.status(400).json({ error: 'max_tokens must be a number between 10 and 512' });
        }

        const cleanedPrompt = prompt.toLowerCase().trim();
        if (['hello', 'hi', 'hey', 'greetings', 'how are you'].includes(cleanedPrompt)) {
            const response = {
                text: `Hey, I’m CASI, doing great! What’s on your mind?`,
                confidence: 0.99,
                outputId: crypto.randomBytes(12).toString('hex'),
                source: 'CASI'
            };
            logger.info(`Generated greeting response for prompt: ${prompt}, outputId: ${response.outputId}`);
            return res.json(response);
        }

        const patterns = app.locals.patterns;
        if (!patterns) {
            throw new Error('Patterns collection not initialized');
        }

        const query = cleanedPrompt.includes('casi') || cleanedPrompt.includes('who are you') ? 'self_description' : cleanedPrompt.split(' ').join(' ');
        const existingPatterns = await patterns.find({
            concept: { $regex: query, $options: 'i' },
            confidence: { $gte: 0.95 }
        }).sort({ feedbackScore: -1, confidence: -1, updatedAt: -1 }).limit(depth).toArray();

        let response;
        const adjustedMaxWords = cleanedPrompt.length < 10 ? Math.min(maxWords, 50) : maxWords;
        if (existingPatterns.length > 0) {
            response = await generateResponse({
                prompt,
                diversityFactor,
                depth,
                breadth,
                maxWords: adjustedMaxWords,
                mood,
                patterns: existingPatterns,
                top_k,
                temperature,
                max_tokens
            });
            logger.info(`Generated response with existing patterns for prompt: ${prompt}, outputId: ${response.outputId}, source: ${response.source}`);
        } else {
            logger.info(`No patterns found for "${prompt}", triggering on-demand scrape`);
            await scrapeOnDemand(query, prompt, patterns);

            const newPatterns = await patterns.find({
                concept: { $regex: query, $options: 'i' },
                confidence: { $gte: 0.95 }
            }).sort({ feedbackScore: -1, confidence: -1, updatedAt: -1 }).limit(depth).toArray();

            if (newPatterns.length > 0) {
                response = await generateResponse({
                    prompt,
                    diversityFactor,
                    depth,
                    breadth,
                    maxWords: adjustedMaxWords,
                    mood,
                    patterns: newPatterns,
                    top_k,
                    temperature,
                    max_tokens
                });
                logger.info(`Generated response with new patterns for prompt: ${prompt}, outputId: ${response.outputId}, source: ${response.source}`);
            } else {
                response = await generateResponse({
                    prompt,
                    diversityFactor,
                    depth,
                    breadth,
                    maxWords: adjustedMaxWords,
                    mood,
                    patterns: [],
                    top_k,
                    temperature,
                    max_tokens
                });
                logger.warn(`No patterns found after on-demand scraping for prompt: ${prompt}`);
            }
        }

        res.json(response);
    } catch (error) {
        const promptForLog = req.body.prompt || 'unknown';
        logger.error(`Generate error for prompt: ${promptForLog}: ${error.message}`);
        res.status(error.status || 500).json({ 
            error: 'Failed to generate response. Please try again.',
            outputId: crypto.randomBytes(12).toString('hex'),
            source: 'Error'
        });
    }
});

app.post('/api/feedback', async (req, res) => {
    try {
        const { outputId, vote } = req.body;
        if (!outputId || !['upvote', 'downvote'].includes(vote)) {
            return res.status(400).json({ error: 'outputId and vote (upvote/downvote) are required' });
        }

        await client.connect();
        const db = client.db('CASIDB');
        const patterns = db.collection('patterns');

        const pattern = await patterns.findOne({ outputId });
        if (!pattern) {
            return res.status(404).json({ error: 'Response not found' });
        }

        const feedbackScore = pattern.feedbackScore || 0;
        const newScore = vote === 'upvote' ? feedbackScore + 0.1 : feedbackScore - 0.1;
        const newConfidence = Math.min(Math.max(pattern.confidence + (vote === 'upvote' ? 0.01 : -0.01), 0.9), 0.995);

        await patterns.updateOne(
            { outputId },
            { $set: { feedbackScore: newScore, confidence: newConfidence, updatedAt: new Date() } }
        );

        logger.info(`Feedback recorded for outputId: ${outputId}, vote: ${vote}, newScore: ${newScore}`);
        res.json({ message: 'Feedback recorded', newScore, newConfidence });
    } catch (error) {
        logger.error(`Feedback error: ${error.message}`);
        res.status(500).json({ error: 'Failed to record feedback' });
    } finally {
        await client.close();
    }
});

app.listen(port, async () => {
    try {
        await init();
        console.log(`Server running on port ${port}`);
    } catch (error) {
        console.error('Server startup failed:', error);
        process.exit(1);
    }
});