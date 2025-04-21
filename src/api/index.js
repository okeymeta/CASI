const express = require('express');
const { MongoClient } = require('mongodb');
const { scrapeAll, scrapeOnDemand } = require('../scrape');
const { generateResponse } = require('../generate');
const { init: initLattice } = require('../lattice');
const winston = require('winston');

const app = express();
const port = process.env.PORT || 3000;
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
            logger.warn('Lattice initialization failed, proceeding without lattice:', error.message);
            const db = client.db('CASIDB');
            app.locals.patterns = db.collection('patterns');
        }

        // Schedule periodic scraping (every 6 hours)
        setInterval(async () => {
            try {
                const results = await scrapeAll('Learn about diverse topics including AI, culture, tech, health, and more');
                logger.info(`Periodic scraping completed, added ${results.length} results`);
            } catch (error) {
                logger.error('Periodic scraping failed:', error.message);
            }
        }, 6 * 60 * 60 * 1000);

        // Initial scrape
        const initialResults = await scrapeAll('Learn about diverse topics including AI, culture, tech, health, and more');
        logger.info(`Initial scraping completed, added ${initialResults.length} results`);
    } catch (error) {
        logger.error('Initialization error:', error);
        throw error;
    }
}

app.post('/api/scrape-all', async (req, res) => {
    try {
        const { prompt } = req.body;
        const results = await scrapeAll(prompt);
        logger.info(`Manual scrape-all completed, added ${results.length} results`);
        res.json(results);
    } catch (error) {
        logger.error('Scrape-all error:', error.message);
        res.status(500).json({ error: error.message });
    }
});

app.post('/api/generate', async (req, res) => {
    try {
        const { prompt, diversityFactor = 0.5, depth = 10, breadth = 5, maxWords = 100, mood = 'neutral' } = req.body;

        if (!prompt) {
            res.status(400).json({ error: 'Prompt is required' });
            return;
        }

        // Ensure patterns collection is available
        const patterns = app.locals.patterns;
        if (!patterns) {
            throw new Error('Patterns collection not initialized');
        }

        // Query existing patterns
        const query = prompt.toLowerCase().split(' ').join(' ');
        const existingPatterns = await patterns.find({
            concept: { $regex: query, $options: 'i' },
            confidence: { $gte: 0.95 }
        }).sort({ updatedAt: -1 }).limit(depth).toArray();

        let response;
        if (existingPatterns.length > 0) {
            // Generate response with existing patterns
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
            // No patterns found, trigger on-demand scraping
            logger.info(`No patterns found for "${prompt}", triggering on-demand scrape`);
            await scrapeOnDemand(query, prompt, patterns);

            // Re-query after scraping
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
                    outputId: Date.now().toString()
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