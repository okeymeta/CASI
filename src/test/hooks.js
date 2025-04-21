const axios = require('axios');
const natural = require('natural');
const lodash = require('lodash');

const API_URL = process.env.API_URL || 'http://localhost:3000';

async function testAccuracy(prompts) {
    const results = await Promise.all(prompts.map(async (prompt) => {
        try {
            const response = await axios.post(`${API_URL}/api/generate`, {
                prompt,
                diversityFactor: 0.5,
                depth: 10,
                breadth: 5,
                maxWords: 100,
                mood: 'neutral'
            }, { timeout: 5000 });

            const verification = await _verifyWithMultipleSources(prompt, response.data.text);
            console.log(`Accuracy test for "${prompt}": ${verification.isAccurate ? 'Passed' : 'Failed'}`);

            return {
                passed: verification.isAccurate,
                score: verification.score,
                prompt,
                output: response.data.text
            };
        } catch (error) {
            console.error(`Accuracy test failed for "${prompt}":`, error);
            return { passed: false, score: 0, prompt, error: error.message };
        }
    }));

    const passed = results.filter(r => r.passed).length;
    return {
        accuracy: passed / results.length,
        passed,
        total: results.length,
        details: results
    };
}

async function _verifyWithMultipleSources(prompt, output) {
    const sources = [
        `https://www.google.com/search?q=${encodeURIComponent(prompt)}`,
        `https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=${encodeURIComponent(prompt)}&format=json`
    ];

    const verifications = await Promise.all(sources.map(async (url) => {
        try {
            const response = await axios.get(url, { timeout: 5000, headers: { 'User-Agent': 'CASI/1.0' } });
            const sourceText = typeof response.data === 'string' ? response.data : JSON.stringify(response.data);
            const similarity = natural.JaroWinklerDistance(prompt, sourceText, { caseSensitive: false });
            return similarity > 0.8;
        } catch (error) {
            console.error(`Verification failed for ${url}:`, error);
            return false;
        }
    }));

    return {
        isAccurate: verifications.some(v => v),
        score: verifications.filter(v => v).length / verifications.length
    };
}

async function testLatency(endpoint, iterations = 100) {
    const times = [];
    for (let i = 0; i < iterations; i++) {
        try {
            const start = Date.now();
            await axios.get(`${API_URL}${endpoint}`, { timeout: 5000 });
            const latency = Date.now() - start;
            times.push(latency);
            console.log(`Latency test ${i + 1}/${iterations}: ${latency}ms`);
        } catch (error) {
            console.error(`Latency test failed:`, error);
            times.push(Infinity);
        }
    }

    const validTimes = times.filter(t => t !== Infinity);
    const avgLatency = validTimes.length ? validTimes.reduce((a, b) => a + b, 0) / validTimes.length : Infinity;
    const maxLatency = validTimes.length ? Math.max(...validTimes) : Infinity;
    return { avgLatency, maxLatency };
}

async function testHallucination(prompts) {
    const results = await Promise.all(prompts.map(async (prompt) => {
        try {
            const response = await axios.post(`${API_URL}/api/generate`, {
                prompt,
                diversityFactor: 0.5,
                depth: 10,
                breadth: 5,
                maxWords: 100,
                mood: 'neutral'
            }, { timeout: 5000 });

            const verification = await axios.get(
                `https://www.google.com/search?q=${encodeURIComponent(response.data.text)}`,
                { timeout: 5000, headers: { 'User-Agent': 'CASI/1.0' } }
            );

            const isValid = verification.status === 200 && verification.data.includes(prompt);
            console.log(`Hallucination test for "${prompt}": ${isValid ? 'Valid' : 'Hallucinated'}`);
            return isValid ? 0 : 1;
        } catch (error) {
            console.error(`Hallucination test failed for "${prompt}":`, error);
            return 1;
        }
    }));

    const failedOutputs = results.map((r, i) => r === 1 ? prompts[i] : null).filter(x => x);
    const rate = results.reduce((a, b) => a + b, 0) / results.length;
    return { rate, failedOutputs };
}

module.exports = {
    testAccuracy,
    testLatency,
    testHallucination
};