const express = require('express');
const { MongoClient } = require('mongodb');
const { scrapeAll, scrapeOnDemand } = require('../scrape');
const { generateResponse, loadPatternsAndTrainModels, initializeModels } = require('../generate');
const { init: initLattice } = require('../lattice');
const winston = require('winston');

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
            console.warn('Proceeding with limited functionality; transformer-based features disabled');
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
        logger.error('Scrape-all error:', error.message);
        res.status(500).json({ error: error.message });
    }
});

app.post('/api/generate', async (req, res) => {
    try {
        const { prompt, diversityFactor = 0.7, depth = 8, breadth = 6, maxWords = 300, mood = 'neutral' } = req.body;

        if (!prompt) {
            return res.status(400).json({ error: 'Prompt is required' });
        }

        const patterns = app.locals.patterns;
        if (!patterns) {
            throw new Error('Patterns collection not initialized');
        }

        const query = prompt.toLowerCase().split(' ').join(' ');
        const existingPatterns = await patterns.find({
            concept: { $regex: query, $options: 'i' },
            confidence: { $gte: 0.95 }
        }).sort({ updatedAt: -1 }).limit(depth).toArray();

        let response;
        if (existingPatterns.length > 0) {
            response = await generateResponse({
                prompt,
                diversityFactor,
                depth,
                breadth,
                maxWords,
                mood,
                patterns: existingPatterns
            });
            logger.info(`Generated response with existing patterns for prompt: ${prompt}, outputId: ${response.outputId}`);
        } else {
            logger.info(`No patterns found for "${prompt}", triggering on-demand scrape`);
            await scrapeOnDemand(query, prompt, patterns);

            const newPatterns = await patterns.find({
                concept: { $regex: query, $options: 'i' },
                confidence: { $gte: 0.95 }
            }).sort({ updatedAt: -1 }).limit(depth).toArray();

            if (newPatterns.length > 0) {
                response = await generateResponse({
                    prompt,
                    diversityFactor,
                    depth,
                    breadth,
                    maxWords,
                    mood,
                    patterns: newPatterns
                });
                logger.info(`Generated response with new patterns for prompt: ${prompt}, outputId: ${response.outputId}`);
            } else {
                response = {
                    text: 'Sorry, I couldn’t find enough info. Try again later!',
                    confidence: 0.95,
                    outputId: crypto.randomBytes(12).toString('hex')
                };
                logger.warn(`No patterns found after on-demand scraping for prompt: ${prompt}`);
            }
        }

        res.json(response);
    } catch (error) {
        logger.error(`Generate error for prompt: ${prompt}: ${error.message}`);
        res.status(error.status || 500).json({ error: error.message });
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