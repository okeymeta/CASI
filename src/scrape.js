const axios = require('axios');
const cheerio = require('cheerio');
const nlp = require('compromise');
const sentiment = require('sentiment');
const { MongoClient } = require('mongodb');
const zlib = require('zlib');
const winston = require('winston');
const xml2js = require('xml2js');
const TfIdf = require('node-tfidf');

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

const RATE_LIMIT_MS = 5000;
const MAX_RETRIES = 5;
const MAX_RESULTS_PER_QUERY = 5;
const ON_DEMAND_MAX_RESULTS = 3;

// User-Agent rotation
const USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64; rv:95.0) Gecko/20100101 Firefox/95.0'
];

const SEARCH_QUERIES = [
    'artificial intelligence',
    'machine learning',
    'data science',
    'blockchain',
    'quantum computing',
    'conversational AI',
    'fintech',
    'education technology',
    'nigerian tech',
    'cybersecurity',
    'cloud computing',
    'internet of things',
    'augmented reality',
    'robotics',
    'yoruba culture',
    'african history',
    'nigerian culture',
    'african literature',
    'cultural heritage',
    'igbo traditions',
    'hausa culture',
    'nigerian festivals',
    'african diaspora',
    'mental health',
    'stress management',
    'global health',
    'nutrition',
    'sleep health',
    'exercise benefits',
    'mindfulness',
    'chronic diseases',
    'public speaking',
    'critical thinking',
    'mathematics',
    'physics',
    'statistics',
    'coding tutorials',
    'stem education',
    'online learning',
    'climate change',
    'sustainable development',
    'renewable energy',
    'biodiversity',
    'urban planning',
    'humor',
    'casual conversation',
    'empathy',
    'social media trends',
    'pop culture',
    'space exploration',
    'genetic engineering',
    'ethical AI',
    'digital nomad lifestyle',
    'ecommerce trends',
    'small talk',
    'internet slang',
    'storytelling',
    'motivational quotes',
    'life advice',
    'friendship dynamics',
    'work-life balance',
    'online banter',
    'meme culture',
    'gaming trends',
    'esports',
    'travel tips',
    'budget travel',
    'foodie culture',
    'cooking hacks',
    'philosophy',
    'existentialism',
    'self-improvement',
    'productivity hacks',
    'fashion trends',
    'sustainable fashion',
    'how are you',
    'tell me a joke',
    'what is your name',
    'what can you do',
    'explain quantum computing simply',
    'how to be happy',
    'how to make friends',
    'how to solve a conflict',
    'how to learn fast',
    'how to be productive',
    'how to deal with stress',
    'how to chat like a human',
    'what is love',
    'what is friendship',
    'how to motivate myself',
    'how to overcome failure',
    'how to start a conversation',
    'how to apologize',
    'how to give advice',
    'how to help someone',
    'how to be creative',
    'how to think critically',
    'how to explain AI to a child',
    'how to explain blockchain to a beginner',
    'how to explain quantum computing to a layman',
    'how to explain machine learning simply',
    'how to explain data science simply',
    'how to explain empathy',
    'how to explain humor',
    'how to explain sadness',
    'how to explain happiness',
    'how to explain curiosity',
    'how to explain intelligence',
    'how to explain consciousness',
    'how to explain ethics',
    'how to explain logic',
    'how to explain reasoning',
    'how to explain creativity',
    'how to explain problem solving',
    'how to explain learning',
    'how to explain memory',
    'how to explain perception',
    'how to explain language',
    'how to explain communication',
    'how to explain understanding',
    'how to explain knowledge',
    'how to explain wisdom',
    'how to explain decision making',
    'how to explain planning',
    'how to explain goal setting',
    'how to explain self-improvement',
    'how to explain self-awareness',
    'how to explain emotional intelligence',
    'how to explain artificial general intelligence',
    'how to explain AGI',
    'how to explain SLM',
    'how to explain LLM',
    'how to explain neural networks',
    'how to explain deep learning',
    'how to explain reinforcement learning',
    'how to explain supervised learning',
    'how to explain unsupervised learning',
    'how to explain transfer learning',
    'how to explain natural language processing',
    'how to explain computer vision',
    'how to explain robotics',
    'how to explain automation',
    'how to explain digital transformation',
    'how to explain the future of AI',
    'how to explain the risks of AI',
    'how to explain the benefits of AI',
    'how to explain the limitations of AI',
    'how to explain the ethics of AI',
    'how to explain the impact of AI on society',
    'how to explain the impact of AI on jobs',
    'how to explain the impact of AI on education',
    'how to explain the impact of AI on healthcare',
    'how to explain the impact of AI on business',
    'how to explain the impact of AI on creativity',
    'how to explain the impact of AI on communication',
    'how to explain the impact of AI on relationships',
    'how to explain the impact of AI on culture',
    'how to explain the impact of AI on the world'
];

const SEARCH_ENGINES = [
    { name: 'Google', url: 'https://www.google.com/search?q=', selector: '.tF2Cxc', title: 'h3', snippet: '.VwiC3b', link: 'a' },
    { name: 'Bing', url: 'https://www.bing.com/search?q=', selector: '.b_algo', title: 'h2', snippet: '.b_caption p', link: 'a' },
    { name: 'DuckDuckGo', url: 'https://duckduckgo.com/?q=', selector: '.result__body', title: '.result__title', snippet: '.result__snippet', link: '.result__url' }
];

const WIKIPEDIA_API = 'https://en.wikipedia.org/w/api.php?action=query&list=search&format=json&srsearch=';
const GOOGLE_NEWS_RSS = 'https://news.google.com/rss/search?q=';
const REDDIT_API = 'https://www.reddit.com/r/all/search.json?q=';
const TIME_AND_DATE = 'https://www.timeanddate.com/worldclock/';

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

// Utility: Remove personal emails and phone numbers, allow business emails
function filterSensitiveInfo(text) {
    // Remove phone numbers (simple patterns)
    text = text.replace(/(\+?\d{1,3}[-.\s]?)?(\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{4,}/g, '[filtered]');
    // Remove personal emails (gmail, yahoo, outlook, protonmail, icloud, hotmail, etc.)
    text = text.replace(/\b[A-Za-z0-9._%+-]+@(gmail|yahoo|outlook|hotmail|protonmail|icloud|aol|mail|zoho|gmx|yandex|qq|163|126|sina|yeah|foxmail|googlemail)\.[A-Za-z]{2,}\b/gi, '[filtered]');
    return text;
}

// Utility: Clean up nbsp, camelCase, and punctuation issues for more human-like sentences
function cleanAIText(text) {
    // Replace HTML entities like nbsp and &nbsp;
    text = text.replace(/&nbsp;|nbsp;/gi, ' ');
    // Remove any remaining HTML entities
    text = text.replace(/&[a-z]+;/gi, ' ');
    // Split camelCase words (e.g., MindfulnessActivity -> Mindfulness Activity)
    text = text.replace(/([a-z])([A-Z])/g, '$1 $2');
    // Remove repeated words (e.g., "CancernbspnbspCancer" -> "Cancer")
    text = text.replace(/\b(\w+)(?:\s+\1\b)+/gi, '$1');
    // Remove trailing source attributions (e.g., "Live Science", "Cancer Network") if at end
    text = text.replace(/([A-Za-z\s]+)\s*(I understand this might be tough\. I'm here to help\.)?$/i, (match, p1, p2) => {
        // Remove if it's a known news/magazine/network source and not a normal sentence
        if (/network|science|news|magazine|times|journal|review|today|daily|tribune|chronicle|gazette|observer|post|reporter|herald|press|bulletin|mirror|standard|star|sun|telegraph|record|register|dispatch|globe|world|business|insider|fortune|forbes|bloomberg|reuters|cnn|bbc|al jazeera|fox|nbc|cbs|abc|ap|upi|wired|the verge|the guardian|the economist|nature|cell|lancet|sciencedaily|sciencenews|sciencemag|arxiv/i.test(p1.trim())) {
            return p2 ? p2 : '';
        }
        return match;
    });
    // Remove double spaces
    text = text.replace(/\s{2,}/g, ' ');
    // Remove stray punctuation at the end
    text = text.replace(/[,;:\-]+$/, '');
    // Ensure proper punctuation at end
    text = text.trim();
    if (text && !/[.!?]$/.test(text)) text += '.';
    // Capitalize first letter
    text = text.charAt(0).toUpperCase() + text.slice(1);
    return text;
}

async function fetchWithRetry(url, retries = MAX_RETRIES) {
    try {
        new URL(url);
    } catch (error) {
        logger.error(`Invalid URL: ${url}`);
        return null;
    }

    const userAgent = USER_AGENTS[Math.floor(Math.random() * USER_AGENTS.length)];
    try {
        const response = await axios.get(url, {
            headers: {
                'User-Agent': userAgent,
                'Accept': 'text/html,application/json,application/xml',
                'Accept-Language': 'en-US,en;q=0.9',
                'Referer': 'https://www.google.com/'
            },
            timeout: 7000
        });
        return response.data;
    } catch (error) {
        if (error.response?.status === 429) {
            logger.warn(`Rate limit hit for ${url}, waiting and retrying`);
            await delay(10000);
            return retries > 0 ? fetchWithRetry(url, retries - 1) : null;
        }
        if (error.response?.status === 403 && retries > 0) {
            logger.warn(`403 Forbidden for ${url}, skipping after ${retries} retries`);
            return null;
        }
        if (retries > 0) {
            logger.warn(`Retrying ${url}, attempts left: ${retries}`);
            await delay(1000 * (MAX_RETRIES - retries + 1));
            return fetchWithRetry(url, retries - 1);
        }
        logger.error(`Failed to fetch ${url}: ${error.message}`);
        return null;
    }
}

function preprocessContent(content) {
    content = content
        .replace(/039/g, "'")
        .replace(/[\u2013\u2014]/g, '-')
        .replace(/’/g, "'")
        .replace(/[^\w\s.,!?']/g, '')
        .replace(/\.\.+/g, '.')
        .trim();
    content = filterSensitiveInfo(content);
    content = cleanAIText(content);
    return content;
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
        logger.warn(`Entity extraction failed for content: ${content.slice(0, 50)}...: ${error.message}`);
        return [];
    }
}

async function scrapeWikipedia(query, prompt, patterns) {
    try {
        const url = `${WIKIPEDIA_API}${encodeURIComponent(query)}`;
        const data = await fetchWithRetry(url);
        if (!data) return [];

        const results = data.query?.search || [];
        const scrapeResults = [];
        for (const result of results.slice(0, MAX_RESULTS_PER_QUERY)) {
            let content = result.snippet.replace(/<\/?[^>]+(>|$)/g, '') || 'No content available';
            if (content === 'No content available' || typeof content !== 'string' || content.trim().length < 5) {
                logger.warn(`Invalid content for Wikipedia article: ${result.title}`);
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
                logger.warn(`Sentiment analysis failed for Wikipedia article: ${result.title}`);
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

                logger.info(`Learned from Wikipedia: ${result.title}, nodes added: ${nodesAdded.insertedId}`);
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
                    logger.warn(`Duplicate entry for Wikipedia: ${result.title}, concept: ${query}`);
                    continue;
                }
                logger.error(`Failed to insert Wikipedia result: ${result.title}: ${error.message}`);
            }
        }

        await delay(RATE_LIMIT_MS);
        return scrapeResults;
    } catch (error) {
        logger.error(`Failed to scrape Wikipedia for query "${query}": ${error.message}`);
        return [];
    }
}

async function scrapeGoogleNews(query, prompt, patterns, maxResults = MAX_RESULTS_PER_QUERY) {
    try {
        const url = `${GOOGLE_NEWS_RSS}${encodeURIComponent(query)}&hl=en-US&gl=US&ceid=US:en`;
        const xml = await fetchWithRetry(url);
        if (!xml) return [];

        const data = await xml2js.parseStringPromise(xml);
        const items = data.rss.channel[0].item || [];
        const scrapeResults = [];
        for (const item of items.slice(0, maxResults)) {
            const title = item.title?.[0] || 'No title';
            let content = item.description?.[0]?.replace(/<\/?[^>]+(>|$)/g, '') || 'No content available';
            const link = item.link?.[0] || '';

            if (content === 'No content available' || typeof content !== 'string' || content.trim().length < 5) {
                logger.warn(`Invalid content for Google News: ${title}`);
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
                logger.warn(`Sentiment analysis failed for Google News: ${title}`);
            }

            try {
                const nodesAdded = await patterns.insertOne({
                    url: link,
                    prompt,
                    entities,
                    sentiment: sentimentScore,
                    content: compressedContent,
                    concept: query,
                    source: 'Google News',
                    title,
                    createdAt: new Date(),
                    updatedAt: new Date(),
                    confidence: Math.min(0.95 + sentimentScore / 100, 0.98)
                });

                logger.info(`Learned from Google News: ${title}, nodes added: ${nodesAdded.insertedId}`);
                scrapeResults.push({
                    status: 'learned',
                    nodesAdded: 1,
                    confidence: Math.min(0.95 + sentimentScore / 100, 0.98),
                    source: 'Google News',
                    title,
                    url: link
                });
            } catch (error) {
                if (error.code === 11000) {
                    logger.warn(`Duplicate entry for Google News: ${title}, concept: ${query}`);
                    continue;
                }
                logger.error(`Failed to insert Google News result: ${title}: ${error.message}`);
            }
        }

        await delay(RATE_LIMIT_MS);
        return scrapeResults;
    } catch (error) {
        logger.error(`Failed to scrape Google News for query "${query}": ${error.message}`);
        return [];
    }
}

async function scrapeReddit(query, prompt, patterns, maxResults = MAX_RESULTS_PER_QUERY) {
    try {
        const url = `${REDDIT_API}${encodeURIComponent(query)}&restrict_sr=on&sort=relevance&t=all`;
        const data = await fetchWithRetry(url);
        if (!data || !data.data || !data.data.children) return [];

        const results = [];
        for (const post of data.data.children.slice(0, maxResults)) {
            const title = post.data.title || 'No title';
            const snippet = post.data.selftext || post.data.description || 'No content available';
            const link = `https://www.reddit.com${post.data.permalink}`;
            if (title && snippet && link && !snippet.includes('promoted') && !post.data.is_video) {
                results.push({ title, snippet, link });
            }
        }

        const scrapeResults = [];
        for (const { title, snippet, link } of results) {
            let content = snippet || 'No content available';
            if (content === 'No content available' || typeof content !== 'string' || content.trim().length < 5) {
                logger.warn(`Invalid content for Reddit post: ${title}`);
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
                logger.warn(`Sentiment analysis failed for Reddit post: ${title}`);
            }

            try {
                const nodesAdded = await patterns.insertOne({
                    url: link,
                    prompt,
                    entities,
                    sentiment: sentimentScore,
                    content: compressedContent,
                    concept: query,
                    source: 'Reddit',
                    title,
                    createdAt: new Date(),
                    updatedAt: new Date(),
                    confidence: Math.min(0.95 + sentimentScore / 100, 0.98)
                });

                logger.info(`Learned from Reddit: ${title}, nodes added: ${nodesAdded.insertedId}`);
                scrapeResults.push({
                    status: 'learned',
                    nodesAdded: 1,
                    confidence: Math.min(0.95 + sentimentScore / 100, 0.98),
                    source: 'Reddit',
                    title,
                    url: link
                });
            } catch (error) {
                if (error.code === 11000) {
                    logger.warn(`Duplicate entry for Reddit: ${title}, concept: ${query}`);
                    continue;
                }
                logger.error(`Failed to insert Reddit result: ${title}: ${error.message}`);
            }
        }

        await delay(RATE_LIMIT_MS);
        return scrapeResults;
    } catch (error) {
        logger.error(`Failed to scrape Reddit for query "${query}": ${error.message}`);
        return [];
    }
}

async function scrapeTimeAndDate(query, prompt, patterns) {
    try {
        const url = TIME_AND_DATE;
        const html = await fetchWithRetry(url);
        if (!html) return [];

        const $ = cheerio.load(html);
        let content = $('#wt-tz').text().trim() || 'No time available';
        if (content === 'No time available' || typeof content !== 'string' || content.trim().length < 5) {
            logger.warn(`Invalid time content from Time and Date`);
            return [];
        }

        content = preprocessContent(content);
        const compressedContent = zlib.gzipSync(content).toString('base64');
        const entities = extractEntities(content);

        let sentimentScore = 0;
        try {
            const sentimentResult = sentiment(content);
            sentimentScore = sentimentResult && typeof sentimentResult.score === 'number' ? sentimentResult.score : 0;
        } catch (error) {
            logger.warn(`Sentiment analysis failed for Time and Date`);
        }

        const title = 'Current Time and Date';
        try {
            const nodesAdded = await patterns.insertOne({
                url,
                prompt,
                entities,
                sentiment: sentimentScore,
                content: compressedContent,
                concept: 'current time',
                source: 'Time and Date',
                title,
                createdAt: new Date(),
                updatedAt: new Date(),
                confidence: 0.98
            });

            logger.info(`Learned from Time and Date: ${title}, nodes added: ${nodesAdded.insertedId}`);
            return [{
                status: 'learned',
                nodesAdded: 1,
                confidence: 0.98,
                source: 'Time and Date',
                title,
                url
            }];
        } catch (error) {
            if (error.code === 11000) {
                logger.warn(`Duplicate entry for Time and Date: ${title}, concept: current time`);
                return [];
            }
            logger.error(`Failed to insert Time and Date result: ${error.message}`);
            return [];
        }
    } catch (error) {
        logger.error(`Failed to scrape Time and Date: ${error.message}`);
        return [];
    }
}

async function scrapeGoogleTopResults(query, prompt, patterns, maxResults = MAX_RESULTS_PER_QUERY) {
    try {
        const searchUrl = `https://www.google.com/search?q=${encodeURIComponent(query)}`;
        const html = await fetchWithRetry(searchUrl);
        if (!html) return [];

        const $ = cheerio.load(html);
        const links = [];
        $('.tF2Cxc a').each((i, element) => {
            if (i < maxResults) {
                const link = $(element).attr('href');
                if (link && link.startsWith('http') && !link.includes('google.com')) {
                    links.push(link);
                }
            }
        });

        const scrapeResults = [];
        for (const link of links) {
            const html = await fetchWithRetry(link);
            if (!html) continue;

            const $ = cheerio.load(html);
            let content = $('.article-body, .post-content, p').text().trim() || 'No content available';
            if (content === 'No content available' || typeof content !== 'string' || content.trim().length < 5) {
                logger.warn(`Invalid content for Google top result: ${link}`);
                continue;
            }

            content = preprocessContent(content.slice(0, 1000));
            const compressedContent = zlib.gzipSync(content).toString('base64');
            const entities = extractEntities(content);

            let sentimentScore = 0;
            try {
                const sentimentResult = sentiment(content);
                sentimentScore = sentimentResult && typeof sentimentResult.score === 'number' ? sentimentResult.score : 0;
            } catch (error) {
                logger.warn(`Sentiment analysis failed for Google top result: ${link}`);
            }

            const title = $('h1').text().trim() || 'Untitled';
            try {
                const nodesAdded = await patterns.insertOne({
                    url: link,
                    prompt,
                    entities,
                    sentiment: sentimentScore,
                    content: compressedContent,
                    concept: query,
                    source: 'Google Top Result',
                    title,
                    createdAt: new Date(),
                    updatedAt: new Date(),
                    confidence: 0.98
                });

                logger.info(`Learned from Google Top Result: ${title}, nodes added: ${nodesAdded.insertedId}`);
                scrapeResults.push({
                    status: 'learned',
                    nodesAdded: 1,
                    confidence: 0.98,
                    source: 'Google Top Result',
                    title,
                    url: link
                });
            } catch (error) {
                if (error.code === 11000) {
                    logger.warn(`Duplicate entry for Google Top Result: ${title}, concept: ${query}`);
                    continue;
                }
                logger.error(`Failed to insert Google Top Result: ${title}: ${error.message}`);
            }

            await delay(RATE_LIMIT_MS);
        }

        return scrapeResults;
    } catch (error) {
        logger.error(`Failed to scrape Google Top Results for query "${query}": ${error.message}`);
        return [];
    }
}

async function scrapeSearchResults(query, prompt, engine, patterns, maxResults = MAX_RESULTS_PER_QUERY) {
    try {
        const searchUrl = `${engine.url}${encodeURIComponent(query)}`;
        const html = await fetchWithRetry(searchUrl);
        if (!html) return [];

        const $ = cheerio.load(html);
        const results = [];
        $(engine.selector).each((i, element) => {
            if (i < maxResults) {
                const title = $(element).find(engine.title).text().trim();
                const snippet = $(element).find(engine.snippet).text().trim();
                const link = $(element).find(engine.link).attr('href');
                if (title && snippet && link && !snippet.includes('Advertisement') && !snippet.includes('Sign up')) {
                    results.push({ title, snippet, link });
                }
            }
        });

        const scrapeResults = [];
        for (const { title, snippet, link } of results) {
            let content = snippet || 'No content available';
            if (content === 'No content available' || typeof content !== 'string' || content.trim().length < 5) {
                logger.warn(`Invalid content for ${engine.name} result: ${title}`);
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
                logger.warn(`Sentiment analysis failed for ${engine.name} result: ${title}`);
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

                logger.info(`Learned from ${engine.name}: ${title}, nodes added: ${nodesAdded.insertedId}`);
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
                    logger.warn(`Duplicate entry for ${engine.name}: ${title}, concept: ${query}`);
                    continue;
                }
                logger.error(`Failed to insert ${engine.name} result: ${title}: ${error.message}`);
            }
        }

        await delay(RATE_LIMIT_MS);
        return scrapeResults;
    } catch (error) {
        logger.error(`Failed to scrape ${engine.name} for query "${query}": ${error.message}`);
        return [];
    }
}

async function scrapeOkeyMetaAI(query, prompt, patterns) {
    logger.info(`[OkeyMetaAI] Called with query: "${query}"`);
    try {
        const apiUrl = 'https://api.okeymeta.com.ng/api/ssailm/model/okeyai3.0-vanguard/okeyai';
        const params = { input: query };
        const response = await axios.get(apiUrl, { params, timeout: 60000 });

        let aiData = response.data;
        if (typeof aiData === 'string') {
            try {
                aiData = JSON.parse(aiData);
            } catch {
                logger.warn(`OkeyMetaAI returned non-JSON string for query: ${query}. Raw: ${aiData}`);
                return [];
            }
        }

        const contentRaw = aiData && (aiData.response || aiData?.model?.response);
        if (!contentRaw || typeof contentRaw !== 'string' || contentRaw.trim().length < 5) {
            logger.warn(`OkeyMetaAI returned no usable response for query: ${query}. Raw response: ${JSON.stringify(response.data)}`);
            return [];
        }

        let content = contentRaw.trim();
        content = filterSensitiveInfo(content);
        content = cleanAIText(content);

        // Distill OkeyAI response using node-tfidf and nlp to create CASI's own sentence
        const tfidf = new TfIdf();
        tfidf.addDocument(content);
        let keywords = [];
        tfidf.listTerms(0).slice(0, 10).forEach(item => keywords.push(item.term));
        let summary = '';
        if (keywords.length > 0) {
            summary = keywords.slice(0, 5).map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(', ') + '. ';
        }
        const nlpDoc = nlp(content);
        const sentences = nlpDoc.sentences().out('array');
        let distilled = summary + (sentences[0] || content);
        distilled = distilled.replace(/\s{2,}/g, ' ').trim();
        if (!/[.!?]$/.test(distilled)) distilled += '.';

        const compressedContent = zlib.gzipSync(content).toString('base64');
        const compressedDistilled = zlib.gzipSync(distilled).toString('base64');
        const entities = [];
        let sentimentScore = 0;
        try {
            const sentimentResult = sentiment(content);
            sentimentScore = sentimentResult && typeof sentimentResult.score === 'number' ? sentimentResult.score : 0;
        } catch {}
        try {
            // Store both raw and distilled
            await patterns.insertOne({
                url: `${apiUrl}?input=${encodeURIComponent(query)}`,
                prompt,
                entities,
                sentiment: sentimentScore,
                content: compressedContent,
                distilled: compressedDistilled,
                concept: query,
                source: 'OkeyMetaAI',
                title: `OkeyMetaAI: ${query}`,
                createdAt: new Date(),
                updatedAt: new Date(),
                confidence: Math.min(0.97 + sentimentScore / 100, 0.99)
            });
            logger.info(`[OkeyMetaAI] Learned and distilled: ${distilled}`);
            return [{
                status: 'learned',
                nodesAdded: 1,
                confidence: Math.min(0.97 + sentimentScore / 100, 0.99),
                source: 'OkeyMetaAI',
                title: `OkeyMetaAI: ${query}`,
                url: `${apiUrl}?input=${encodeURIComponent(query)}`,
                distilled
            }];
        } catch (error) {
            if (error.code === 11000) {
                logger.info(`OkeyMetaAI duplicate for query: ${query}`);
                return [{
                    status: 'duplicate',
                    nodesAdded: 0,
                    confidence: Math.min(0.97 + sentimentScore / 100, 0.99),
                    source: 'OkeyMetaAI',
                    title: `OkeyMetaAI: ${query}`,
                    url: `${apiUrl}?input=${encodeURIComponent(query)}`
                }];
            }
            logger.error(`Failed to insert OkeyMetaAI result for query: ${query}: ${error.message}`);
            return [];
        }
    } catch (error) {
        logger.error(`OkeyMetaAI error for query: ${query}: ${error.message}`);
        return [];
    }
}

async function scrapeOnDemand(query, prompt, patterns) {
    try {
        logger.info(`On-demand scraping for new query: ${query}`);

        const results = [];

        // Scrape from OkeyMeta AI API for conversational data FIRST
        const aiResults = await scrapeOkeyMetaAI(query, prompt, patterns);
        results.push(...aiResults);

        // Scrape Google
        const googleResults = await scrapeSearchResults(query, prompt, SEARCH_ENGINES[0], patterns, ON_DEMAND_MAX_RESULTS);
        results.push(...googleResults);

        // Scrape Reddit
        const redditResults = await scrapeReddit(query, prompt, patterns, ON_DEMAND_MAX_RESULTS);
        results.push(...redditResults);

        // Scrape Google News
        const newsResults = await scrapeGoogleNews(query, prompt, patterns, ON_DEMAND_MAX_RESULTS);
        results.push(...newsResults);

        // Scrape Google Top Results
        const topResults = await scrapeGoogleTopResults(query, prompt, patterns, ON_DEMAND_MAX_RESULTS);
        results.push(...topResults);

        logger.info(`On-demand scraping completed for "${query}", added ${results.length} results`);
        return results;
    } catch (error) {
        logger.error(`On-demand scraping failed for "${query}": ${error.message}`);
        return [];
    }
}

async function scrapeAll(prompt = 'Learn about diverse topics including AI, culture, tech, health, and more') {
    try {
        await client.connect();
        const db = client.db('CASIDB');
        const patterns = db.collection('patterns');
        await dropIndexIfExists(patterns, 'concept_1');

        // Create indexes for faster queries
        await patterns.createIndex({ concept: 1, updatedAt: -1 });
        await patterns.createIndex({ source: 1 });
        await patterns.createIndex({ entities: 1 });

        const results = [];
        for (const query of SEARCH_QUERIES) {
            logger.info(`Scraping for query: ${query}`);

            // Scrape from OkeyMeta AI API for conversational data FIRST
            const aiResults = await scrapeOkeyMetaAI(query, prompt, patterns);
            results.push(...aiResults);

            // Only scrape other sources if OkeyMetaAI returned no usable result
            if (!aiResults || aiResults.length === 0) {
                // Scrape Wikipedia
                const wikiResults = await scrapeWikipedia(query, prompt, patterns);
                results.push(...wikiResults);

                // Scrape Google News
                const newsResults = await scrapeGoogleNews(query, prompt, patterns);
                results.push(...newsResults);
            }

            // Scrape Reddit
            const redditResults = await scrapeReddit(query, prompt, patterns);
            results.push(...redditResults);

            // Scrape Google Top Results
            const topResults = await scrapeGoogleTopResults(query, prompt, patterns);
            results.push(...topResults);

            // Scrape Google, Bing, DuckDuckGo
            const searchPromises = SEARCH_ENGINES.map(engine => scrapeSearchResults(query, prompt, engine, patterns));
            const searchResults = await Promise.all(searchPromises);
            searchResults.forEach(engineResults => results.push(...engineResults));

            // Scrape Time and Date (only for 'current time')
            if (query === 'current time') {
                const timeResults = await scrapeTimeAndDate(query, prompt, patterns);
                results.push(...timeResults);
            }

            await delay(RATE_LIMIT_MS);
        }

        logger.info(`Scraping completed, added ${results.length} results`);
        return results;
    } catch (error) {
        logger.error(`Scrape-all error: ${error.message}`);
        throw error;
    } finally {
        await client.close();
    }
}

module.exports = { scrapeAll, scrapeOnDemand };