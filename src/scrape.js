const cheerio = require('cheerio');
const axios = require('axios');
const winkNLP = require('wink-nlp');
const model = require('wink-eng-lite-web-model');
const nlp = winkNLP(model);
const zlib = require('zlib');
const Sentiment = require('sentiment');
const lattice = require('./lattice');
const lodash = require('lodash');

class WebScraper {
    constructor() {
        this.rateLimits = {
            google: { requests: 0, lastReset: Date.now() },
            reddit: { requests: 0, lastReset: Date.now() },
            wikipedia: { requests: 0, lastReset: Date.now() }
        };
        this.sentiment = new Sentiment();
    }

    async scrapeAndLearn(url, prompt) {
        const site = this._getSiteFromUrl(url);
        if (!this._checkRateLimit(site)) {
            throw { error: 'Rate limit exceeded', status: 429, retryAfter: 60, timestamp: new Date() };
        }

        try {
            const robotsUrl = new URL('/robots.txt', url).toString();
            await axios.head(robotsUrl, { timeout: 3000, headers: { 'User-Agent': 'CASI/1.0' } });

            const response = await this._retryRequest(url);
            const $ = cheerio.load(response.data);
            const text = $('p').text();
            if (!text) {
                throw { error: 'No content found', status: 404, timestamp: new Date() };
            }

            const doc = nlp.readDoc(text);
            const patterns = this._extractPatterns(doc, prompt);
            if (patterns.length === 0) {
                console.warn(`No patterns extracted from ${url}`);
                return { status: 'learned', patterns: 0, timestamp: new Date(), siteCompliance: true };
            }

            for (const pattern of patterns) {
                await lattice.createNode({
                    concept: pattern.concept,
                    emotions: pattern.emotions,
                    tone: pattern.tone,
                    confidence: pattern.confidence,
                    verified: false
                });
                if (pattern.related) {
                    for (const rel of pattern.related) {
                        await lattice.createEdge({
                            source: pattern.concept,
                            target: rel,
                            strength: 0.9
                        });
                    }
                }
            }

            console.log(`Scraped ${url}, learned ${patterns.length} patterns`);
            return {
                status: 'learned',
                patterns: patterns.length,
                timestamp: new Date(),
                siteCompliance: true
            };
        } catch (error) {
            console.error(`Scrape failed for ${url}:`, error);
            throw {
                error: error.message || 'Scraping failed',
                status: error.status || 500,
                retryAfter: error.retryAfter,
                timestamp: new Date()
            };
        }
    }

    async _retryRequest(url, maxRetries = 3) {
        for (let i = 0; i < maxRetries; i++) {
            try {
                return await axios.get(url, {
                    headers: { 'User-Agent': 'CASI/1.0 (compliance@casi.ai)' },
                    timeout: 5000
                });
            } catch (error) {
                if (error.response?.status === 429) {
                    const delay = Math.pow(2, i) * 30000;
                    console.warn(`429 error, retrying after ${delay}ms`);
                    await new Promise(resolve => setTimeout(resolve, delay));
                    continue;
                }
                throw error;
            }
        }
        throw { error: 'Max retries exceeded', status: 429, retryAfter: 60, timestamp: new Date() };
    }

    _getSiteFromUrl(url) {
        if (url.includes('google.com')) return 'google';
        if (url.includes('reddit.com')) return 'reddit';
        if (url.includes('wikipedia.org')) return 'wikipedia';
        throw { error: 'Unsupported site', status: 400, timestamp: new Date() };
    }

    _checkRateLimit(site) {
        const limit = this.rateLimits[site];
        const now = Date.now();
        if (now - limit.lastReset > 1000) {
            limit.requests = 0;
            limit.lastReset = now;
        }
        if (limit.requests >= 1) return false;
        limit.requests++;
        return true;
    }

    _extractPatterns(doc, prompt) {
        const entities = doc.entities().out();
        const patterns = [];
        for (const entity of entities) {
            const text = lodash.escape(entity);
            if (text.length < 1 || text.length > 100) continue;
            const analysis = this.sentiment.analyze(text);
            const tone = analysis.score > 0 ? 'playful' : analysis.score < 0 ? 'empathetic' : 'neutral';
            patterns.push({
                concept: text,
                emotions: this._detectEmotions(text),
                tone,
                confidence: 0.9,
                source: prompt,
                related: doc.tokens().out().filter(t => t !== text).slice(0, 5)
            });
        }
        return patterns;
    }

    _detectEmotions(text) {
        const doc = nlp.readDoc(text);
        return doc.tokens().filter(t => t.out(nlp.its.type) === 'emotion').out();
    }
}

module.exports = new WebScraper();