const axios = require('axios');
const cheerio = require('cheerio');
const winkNLP = require('wink-nlp');
const model = require('wink-eng-lite-web-model');
const nlp = winkNLP(model);
const sentiment = require('sentiment');
const { MongoClient } = require('mongodb');
const zlib = require('zlib');

const mongoUri = 'mongodb+srv://nwaozor:nwaozor@cluster0.rmvi7qm.mongodb.net/CASIDB?retryWrites=true&w=majority';
const client = new MongoClient(mongoUri);

// Comprehensive list of authoritative sites for CASI's knowledge base
const SCRAPE_URLS = [
    // Science and Technology
    'https://www.nature.com/articles?sort=pub-date&year=2025',
    'https://www.technologyreview.com/category/artificial-intelligence/',
    'https://www.scientificamerican.com/section/technology/',
    // Psychology and Well-being
    'https://www.psychologytoday.com/us/basics/stress',
    'https://www.helpguide.org/articles/stress/stress-management.htm',
    'https://www.mayoclinic.org/healthy-lifestyle/stress-management/in-depth/stress-relief/art-20044476',
    // Education and Skills
    'https://www.mindtools.com/a05krsj/delivering-great-presentations',
    'https://www.khanacademy.org/college-careers-more/learnstorm-growth-mindset-activities-us',
    'https://www.coursera.org/articles/public-speaking',
    // Culture and Society
    'https://www.bbc.com/culture',
    'https://en.unesco.org/themes/intercultural-dialogue',
    'https://www.nationalgeographic.com/culture',
    // AI and Ethics
    'https://x.ai/blog', // xAI blog for AI alignment
    'https://hai.stanford.edu/research/ethics-and-society',
    'https://www.aiethicist.org/blog',
    // General Knowledge
    'https://en.wikipedia.org/wiki/Main_Page', // Targeted pages can be added
    'https://www.britannica.com/topic/artificial-intelligence'
];

// Rate limiting configuration
const RATE_LIMIT_MS = 5000; // 5 seconds between requests
const MAX_RETRIES = 3;

async function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function fetchWithRetry(url, retries = MAX_RETRIES) {
    try {
        const response = await axios.get(url, {
            headers: {
                'User-Agent': 'CASI-WebScraper/1.0 (+https://github.com/okeymeta/CASI)',
                'Accept': 'text/html'
            },
            timeout: 10000
        });
        return response.data;
    } catch (error) {
        if (retries > 0) {
            console.warn(`Retrying ${url}, attempts left: ${retries}`);
            await delay(1000);
            return fetchWithRetry(url, retries - 1);
        }
        throw new Error(`Failed to fetch ${url}: ${error.message}`);
    }
}

async function scrapeAndLearn(url, prompt) {
    try {
        const html = await fetchWithRetry(url);
        const $ = cheerio.load(html);
        const textContent = $('p, h1, h2, h3, li').map((i, el) => $(el).text().trim()).get().join(' ');
        const compressedContent = zlib.gzipSync(textContent).toString('base64');

        const doc = nlp.readDoc(textContent);
        const entities = doc.entities().out(winkNLP.its.detail);
        const sentimentScore = sentiment(textContent).score;

        await client.connect();
        const db = client.db('CASIDB');
        const patterns = db.collection('patterns');

        const nodesAdded = await patterns.insertOne({
            url,
            prompt,
            entities: entities.slice(0, 50),
            sentiment: sentimentScore,
            content: compressedContent,
            concept: prompt,
            updatedAt: new Date(),
            confidence: Math.min(0.95 + sentimentScore / 100, 0.98)
        });

        console.log(`Learned from ${url}, nodes added: ${nodesAdded.insertedId}`);
        await delay(RATE_LIMIT_MS);

        return {
            status: 'learned',
            nodesAdded: 1,
            confidence: Math.min(0.95 + sentimentScore / 100, 0.98)
        };
    } catch (error) {
        console.error(`Error scraping ${url}:`, error);
        throw { message: `Scraping failed: ${error.message}`, status: 500 };
    }
}

async function scrapeAll(prompt) {
    const results = [];
    for (const url of SCRAPE_URLS) {
        try {
            const result = await scrapeAndLearn(url, prompt);
            results.push(result);
            await delay(RATE_LIMIT_MS);
        } catch (error) {
            console.error(`Failed to scrape ${url}:`, error);
        }
    }
    return results;
}

module.exports = {
    scrapeAndLearn,
    scrapeAll
};