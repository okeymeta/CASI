const axios = require('axios');
const cheerio = require('cheerio');
const nlp = require('compromise');
const sentiment = require('sentiment');
const { MongoClient } = require('mongodb');
const zlib = require('zlib');

const mongoUri = 'mongodb+srv://nwaozor:nwaozor@cluster0.rmvi7qm.mongodb.net/CASIDB?retryWrites=true&w=majority';
const client = new MongoClient(mongoUri);

const SEARCH_QUERIES = [
    'artificial intelligence',
    'stress management',
    'public speaking',
    'yoruba culture',
    'nigerian tech',
    'mental health',
    'education technology',
    'african history',
    'sustainable development',
    'data science',
    'blockchain',
    'climate change',
    'global health',
    'fintech',
    'cultural heritage'
];

const SEARCH_ENGINES = [
    { name: 'Google', url: 'https://www.google.com/search?q=', selector: '.g', title: 'h3', snippet: '.VwiC3b', link: 'a' },
    { name: 'Bing', url: 'https://www.bing.com/search?q=', selector: '.b_algo', title: 'h2', snippet: '.b_caption p', link: 'a' }
];

const WIKIPEDIA_API = 'https://en.wikipedia.org/w/api.php?action=query&list=search&format=json&srsearch=';

const RATE_LIMIT_MS = 5000;
const MAX_RETRIES = 5;
const MAX_RESULTS_PER_QUERY = 5;

async function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function dropIndexIfExists(collection, indexName) {
    try {
        await collection.dropIndex(indexName);
        console.log(`Dropped ${indexName} index`);
    } catch (error) {
        if (error.codeName === 'IndexNotFound') {
            console.log(`${indexName} index not found, proceeding`);
        } else {
            console.warn(`Failed to drop ${indexName} index: ${error.message}`);
        }
    }
}

async function fetchWithRetry(url, retries = MAX_RETRIES) {
    try {
        const response = await axios.get(url, {
            headers: {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36',
                'Accept': 'text/html,application/json'
            },
            timeout: 5000
        });
        return response.data;
    } catch (error) {
        if (retries > 0) {
            console.warn(`Retrying ${url}, attempts left: ${retries}`);
            await delay(1000 * (MAX_RETRIES - retries + 1));
            return fetchWithRetry(url, retries - 1);
        }
        console.error(`Failed to fetch ${url}: ${error.message}`);
        return null;
    }
}

function preprocessContent(content) {
    return content
        .replace(/039/g, "'")
        .replace(/[\u2013\u2014]/g, '-')
        .replace(/’/g, "'")
        .replace(/[^\w\s.,!?']/g, '')
        .replace(/\.\.+/g, '.')
        .trim();
}

function extractEntities(content) {
    try {
        const doc = nlp(content);
        const entities = [];
        doc.people().forEach(p => entities.push({ value: p.text(), type: 'person' }));
        doc.places().forEach(p => entities.push({ value: p.text(), type: 'place' }));
        doc.organizations().forEach(o => entities.push({ value: o.text(), type: 'organization' }));
        doc.topics().forEach(t => entities.push({ value: t.text(), type: 'topic' }));
        return entities;
    } catch (error) {
        console.warn(`Entity extraction failed for content: ${content.slice(0, 50)}...`, error.message);
        return [];
    }
}

async function scrapeWikipedia(query, prompt) {
    try {
        const url = `${WIKIPEDIA_API}${encodeURIComponent(query)}`;
        const data = await fetchWithRetry(url);
        if (!data) return [];

        await client.connect();
        const db = client.db('CASIDB');
        const patterns = db.collection('patterns');

        const results = data.query?.search || [];
        const scrapeResults = [];
        for (const result of results.slice(0, MAX_RESULTS_PER_QUERY)) {
            let content = result.snippet.replace(/<\/?[^>]+(>|$)/g, '') || 'No content available';
            if (content === 'No content available' || typeof content !== 'string' || content.trim().length < 5) {
                console.warn(`Invalid content for Wikipedia article: ${result.title}, content: ${content}`);
                continue;
            }

            content = preprocessContent(content);
            const compressedContent = zlib.gzipSync(content).toString('base64');
            const entities = extractEntities(content);

            let sentimentScore = 0;
            try {
                const sentimentResult = sentiment(content);
                sentimentScore = sentimentResult && typeof sentimentResult.score === 'number' ? sentimentResult.score : 0;
            } catch (error) {
                console.warn(`Sentiment analysis failed for Wikipedia article: ${result.title}, content: ${content}`, error.message);
            }

            try {
                const nodesAdded = await patterns.insertOne({
                    url: `https://en.wikipedia.org/wiki/${encodeURIComponent(result.title)}`,
                    prompt,
                    entities,
                    sentiment: sentimentScore,
                    content: compressedContent,
                    concept: query,
                    source: 'Wikipedia',
                    title: result.title,
                    createdAt: new Date(),
                    updatedAt: new Date(),
                    confidence: Math.min(0.95 + sentimentScore / 100, 0.98)
                });

                console.log(`Learned from Wikipedia: ${result.title}, nodes added: ${nodesAdded.insertedId}`);
                scrapeResults.push({
                    status: 'learned',
                    nodesAdded: 1,
                    confidence: Math.min(0.95 + sentimentScore / 100, 0.98),
                    source: 'Wikipedia',
                    title: result.title,
                    url: `https://en.wikipedia.org/wiki/${encodeURIComponent(result.title)}`
                });
            } catch (error) {
                if (error.code === 11000) {
                    console.warn(`Duplicate entry for Wikipedia: ${result.title}, concept: ${query}, continuing`);
                    continue;
                }
                console.error(`Failed to insert Wikipedia result: ${result.title}`, error.message);
            }
        }

        await delay(RATE_LIMIT_MS);
        return scrapeResults;
    } catch (error) {
        console.error(`Failed to scrape Wikipedia for query "${query}":`, error.message);
        return [];
    }
}

async function scrapeSearchResults(query, prompt, engine) {
    try {
        const searchUrl = `${engine.url}${encodeURIComponent(query)}`;
        const html = await fetchWithRetry(searchUrl);
        if (!html) return [];

        await client.connect();
        const db = client.db('CASIDB');
        const patterns = db.collection('patterns');

        const $ = cheerio.load(html);
        const results = [];
        $(engine.selector).each((i, element) => {
            if (i < MAX_RESULTS_PER_QUERY) {
                const title = $(element).find(engine.title).text().trim();
                const snippet = $(element).find(engine.snippet).text().trim();
                const link = $(element).find(engine.link).attr('href');

                if (title && snippet && link) {
                    results.push({ title, snippet, link });
                }
            }
        });

        const scrapeResults = [];
        for (const { title, snippet, link } of results) {
            let content = snippet || 'No content available';
            if (content === 'No content available' || typeof content !== 'string' || content.trim().length < 5) {
                console.warn(`Invalid content for ${engine.name} result: ${title}, content: ${content}`);
                continue;
            }

            content = preprocessContent(content);
            const compressedContent = zlib.gzipSync(content).toString('base64');
            const entities = extractEntities(content);

            let sentimentScore = 0;
            try {
                const sentimentResult = sentiment(content);
                sentimentScore = sentimentResult && typeof sentimentResult.score === 'number' ? sentimentResult.score : 0;
            } catch (error) {
                console.warn(`Sentiment analysis failed for ${engine.name} result: ${title}, content: ${content}`, error.message);
            }

            try {
                const nodesAdded = await patterns.insertOne({
                    url: link,
                    prompt,
                    entities,
                    sentiment: sentimentScore,
                    content: compressedContent,
                    concept: query,
                    source: engine.name,
                    title,
                    createdAt: new Date(),
                    updatedAt: new Date(),
                    confidence: Math.min(0.95 + sentimentScore / 100, 0.98)
                });

                console.log(`Learned from ${engine.name}: ${title}, nodes added: ${nodesAdded.insertedId}`);
                scrapeResults.push({
                    status: 'learned',
                    nodesAdded: 1,
                    confidence: Math.min(0.95 + sentimentScore / 100, 0.98),
                    source: engine.name,
                    title,
                    url: link
                });
            } catch (error) {
                if (error.code === 11000) {
                    console.warn(`Duplicate entry for ${engine.name}: ${title}, concept: ${query}, continuing`);
                    continue;
                }
                console.error(`Failed to insert ${engine.name} result: ${title}`, error.message);
            }
        }

        await delay(RATE_LIMIT_MS);
        return scrapeResults;
    } catch (error) {
        console.error(`Failed to scrape ${engine.name} for query "${query}":`, error.message);
        return [];
    }
}

async function scrapeAll(prompt = 'Learn about diverse topics including AI, culture, tech, health, and more') {
    try {
        await client.connect();
        const db = client.db('CASIDB');
        const patterns = db.collection('patterns');
        await dropIndexIfExists(patterns, 'concept_1');

        const results = [];
        for (const query of SEARCH_QUERIES) {
            console.log(`[ScrapeAll] Processing query: ${query}`);
            // Scrape Wikipedia
            const wikiResults = await scrapeWikipedia(query, prompt);
            results.push(...wikiResults);

            // Scrape Google and Bing
            const searchPromises = SEARCH_ENGINES.map(engine => scrapeSearchResults(query, prompt, engine));
            const searchResults = await Promise.all(searchPromises);
            searchResults.forEach(engineResults => results.push(...engineResults));

            await delay(RATE_LIMIT_MS);
        }
        console.log(`[ScrapeAll] Completed, total results: ${results.length}`);
        return results;
    } catch (error) {
        console.error(`[ScrapeAll] Failed: ${error.message}`);
        return [];
    } finally {
        await client.close();
    }
}

module.exports = {
    scrapeAll
};