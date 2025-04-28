const natural = require('natural');
const sentiment = require('sentiment');
const nlp = require('compromise');
const handlebars = require('handlebars');
const crypto = require('crypto');
const axios = require('axios');
const lodash = require('lodash');
const { MongoClient } = require('mongodb');
const zlib = require('zlib');
const Typo = require('typo-js');
const { NlpManager } = require('node-nlp');
const EventEmitter = require('events');
const { pipeline, env } = require('@xenova/transformers');

// Configure transformers for CPU and local caching
env.allowLocalModels = true;
env.localModelPath = './models';
env.backends.onnx.wasm.numThreads = 1;

const mongoUri = 'mongodb+srv://nwaozor:nwaozor@cluster0.rmvi7qm.mongodb.net/CASIDB?retryWrites=true&w=majority';
const client = new MongoClient(mongoUri);

// Optimize NLP manager for speed
let nlpManager = new NlpManager({ languages: ['en'], forceNER: true, nlu: { epochs: 20, batchSize: 8 } });
const learningEmitter = new EventEmitter();
let patternsCache = [];
let sentenceEncoder = null;
let textGenerator = null;

// K-means clustering (optimized for speed)
function kMeansCluster(embeddings, k = 2) {
    const centroids = embeddings.slice(0, k);
    const clusters = Array(k).fill().map(() => []);
    
    for (let i = 0; i < 3; i++) {
        clusters.forEach(c => c.length = 0);
        embeddings.forEach((emb, idx) => {
            const distances = centroids.map(c => 
                Math.sqrt(lodash.sum(c.map((v, j) => (v - emb[j]) ** 2)))
            );
            const clusterIdx = distances.indexOf(Math.min(...distances));
            clusters[clusterIdx].push(idx);
        });
        centroids.forEach((c, j) => {
            if (clusters[j].length > 0) {
                const mean = embeddings[0].map((_, i) => 
                    lodash.mean(clusters[j].map(idx => embeddings[idx][i]))
                );
                c.splice(0, c.length, ...mean);
            }
        });
    }
    
    return clusters;
}

// Prune low-confidence patterns
async function prunePatterns(maxDocs = 200, minConfidence = 0.9) {
    try {
        await client.connect();
        const db = client.db('CASIDB');
        const patterns = db.collection('patterns');
        const result = await patterns.deleteMany({ confidence: { $lt: minConfidence } });
        console.log(`[CASI] Pruned ${result.deletedCount} low-confidence patterns`);
    } catch (error) {
        console.error('[CASI] Prune patterns error:', error.message);
    } finally {
        await client.close();
    }
}

// Check Hugging Face access
async function checkHuggingFaceAccess() {
    try {
        const response = await axios.head('https://huggingface.co/Xenova/all-MiniLM-L6-v2', { timeout: 5000 });
        console.log('[CASI] Hugging Face accessible:', response.status);
        return true;
    } catch (error) {
        console.error('[CASI] Cannot access Hugging Face:', error.message);
        return false;
    }
}

// Initialize transformer models with fallback
async function initializeModels() {
    try {
        console.log('[CASI] Initializing transformer models...');
        await checkHuggingFaceAccess();

        try {
            sentenceEncoder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', { 
                device: 'cpu',
                cache_dir: './models',
                quantized: true
            });
        } catch (error) {
            console.warn('[CASI] Failed to load Xenova/all-MiniLM-L6-v2:', error.message);
            console.log('[CASI] Falling back to Xenova/distilroberta-base...');
            sentenceEncoder = await pipeline('feature-extraction', 'Xenova/distilroberta-base', { 
                device: 'cpu',
                cache_dir: './models',
                quantized: true
            });
        }

        try {
            textGenerator = await pipeline('text-generation', 'Xenova/distilgpt2', { 
                device: 'cpu',
                cache_dir: './models',
                quantized: true
            });
        } catch (error) {
            console.warn('[CASI] Failed to load Xenova/distilgpt2:', error.message);
            console.log('[CASI] Falling back to Xenova/gpt2...');
            textGenerator = await pipeline('text-generation', 'Xenova/gpt2', { 
                device: 'cpu',
                cache_dir: './models',
                quantized: true
            });
        }

        console.log('[CASI] Transformer models initialized');
    } catch (error) {
        console.error('[CASI] Model initialization failed:', error.message);
        sentenceEncoder = null;
        textGenerator = null;
        console.warn('[CASI] Proceeding without transformer models; using fallback mode');
    }
}

// Compute sentence embeddings with fallback
async function computeEmbeddings(texts) {
    if (!sentenceEncoder) {
        console.warn('[CASI] Sentence encoder not initialized, using zero vectors');
        return texts.map(() => Array(384).fill(0));
    }
    try {
        const embeddings = await Promise.all(texts.map(async text => {
            const output = await sentenceEncoder(text, { pooling: 'mean', normalize: true });
            return Array.from(output.data);
        }));
        return embeddings;
    } catch (error) {
        console.error('[CASI] Compute embeddings error:', error.message);
        return texts.map(() => Array(384).fill(0));
    }
}

// Define Handlebars templates
const templates = {
    list: `{{#each items}}{{index}}. **{{name}}**: {{description}}\n{{/each}}`,
    explanation: `**{{topic}}**: {{description}}. Examples: {{examples}}.`,
    comparison: `**{{item1}}** vs **{{item2}}**:\n- **{{item1}}**: {{desc1}}\n- **{{item2}}**: {{desc2}}\nSummary: {{summary}}.`,
    definition: `**{{term}}** is defined as {{definition}}. Used in: {{context}}.`,
    story: `**{{title}}**\n{{intro}}\n{{body}}\nConclusion: {{conclusion}}.`,
    qa: `**Question**: {{question}}\n**Answer**: {{answer}}.`,
    summary: `**Summary of {{topic}}**: {{overview}}. Key points:\n{{points}}.`,
    table: `| **Item** | **Description** |\n|----------|-----------------|\n{{#each rows}}| {{name}} | {{description}} |\n{{/each}}`,
    tutorial: `**Tutorial: {{topic}}**\n**Objective**: {{objective}}\n**Steps**:\n{{#each steps}}{{index}}. {{step}}\n{{/each}}\n**Tips**: {{tips}}.`,
    pros_cons: `**Pros and Cons of {{topic}}**\n**Pros**:\n{{#each pros}}- {{.}}\n{{/each}}\n**Cons**:\n{{#each cons}}- {{.}}\n{{/each}}\n**Verdict**: {{verdict}}.`,
    faq: `**FAQ: {{topic}}**\n{{#each questions}}{{index}}. **{{question}}**: {{answer}}\n{{/each}}`,
    timeline: `**Timeline of {{topic}}**\n{{#each events}}{{year}}: {{event}}\n{{/each}}`,
    code_snippet: `**{{language}} Code: {{topic}}**\n\`\`\`{{language}}\n{{code}}\n\`\`\`\n**Explanation**: {{explanation}}.`,
    case_study: `**Case Study: {{topic}}**\n**Background**: {{background}}\n**Analysis**: {{analysis}}\n**Outcome**: {{outcome}}.`,
    recommendation: `**Recommendations for {{topic}}**\n{{#each items}}{{index}}. {{item}}\n{{/each}}\n**Rationale**: {{rationale}}.`,
    interview: `**Interview on {{topic}}**\n{{#each questions}}{{index}}. **{{question}}**: {{answer}}\n{{/each}}`,
    glossary: `**Glossary: {{topic}}**\n{{#each terms}}{{term}}: {{definition}}\n{{/each}}`,
    troubleshooting: `**Troubleshooting: {{topic}}**\n**Issues**:\n{{#each issues}}{{index}}. {{issue}}: {{solution}}\n{{/each}}\n**Tips**: {{tips}}.`,
    roadmap: `**Roadmap for {{topic}}**\n**Phases**:\n{{#each phases}}{{index}}. {{phase}} ({{timeline}})\n{{/each}}\n**Goals**: {{goals}}.`,
    analysis: `**Analysis of {{topic}}**\n**Overview**: {{overview}}\n**Findings**:\n{{#each findings}}- {{.}}\n{{/each}}\n**Conclusion**: {{conclusion}}.`,
    fallback: `I'm exploring **{{topic}}**. Here's my take: {{context}}. Let's dive deeper!`
};

async function loadPatternsAndTrainModels(maxDocs = 200) {
    try {
        await client.connect();
        const db = client.db('CASIDB');
        const patterns = db.collection('patterns');
        const allPatterns = await patterns.find({}).limit(maxDocs).toArray();

        nlpManager = new NlpManager({ languages: ['en'], forceNER: true, nlu: { epochs: 20, batchSize: 8 } });
        patternsCache = [];

        let count = 0;
        for (const pattern of allPatterns) {
            let text;
            try {
                text = zlib.gunzipSync(Buffer.from(pattern.content, 'base64')).toString();
            } catch {
                text = '';
            }
            if (text && pattern.concept) {
                nlpManager.addDocument('en', text, pattern.concept);
                nlpManager.addAnswer('en', pattern.concept, text);
                patternsCache.push({ concept: pattern.concept, text, pattern });
                count++;
                if (count >= maxDocs) break;
            }
        }
        await nlpManager.train();

        if (patternsCache.length > 2 && sentenceEncoder) {
            const texts = patternsCache.map(p => p.text);
            const embeddings = await computeEmbeddings(texts);
            const clusters = kMeansCluster(embeddings, 2);
            console.log('[CASI] Clustered patterns:', clusters.map(c => c.length));
        }

        await prunePatterns(maxDocs, 0.9);
    } catch (error) {
        console.error('[CASI] Load patterns error:', error.message);
        throw error;
    } finally {
        await client.close();
    }
}

learningEmitter.on('newPattern', async (pattern) => {
    try {
        let text;
        try {
            text = zlib.gunzipSync(Buffer.from(pattern.content, 'base64')).toString();
        } catch {
            text = '';
        }
        if (text && pattern.concept) {
            nlpManager.addDocument('en', text, pattern.concept);
            nlpManager.addAnswer('en', pattern.concept, text);
            patternsCache.push({ concept: pattern.concept, text, pattern });
            await nlpManager.train();
        }
    } catch (error) {
        console.error('[CASI] New pattern error:', error.message);
    }
});

function preprocessPrompt(prompt) {
    return prompt
        .replace(/039/g, "'")
        .replace(/[\u2013\u2014]/g, '-')
        .replace(/’/g, "'")
        .replace(/[^\w\s.,!?*'-|]/g, '')
        .replace(/\.\.+/g, '.')
        .trim()
        .toLowerCase();
}

function preserveFormatting(text) {
    text = text
        .replace(/(\n\s*[-*]\s*)/g, '\n$1')
        .replace(/(\n\s*\d+\.\s*)/g, '\n$1')
        .replace(/(\*\*[^\*]+\*\*)/g, '$1')
        .replace(/(\*[^\*]+\*)/g, '$1')
        .replace(/(^\s*\|.*\|\s*$)/gm, '$1');
    text = text.replace(/\n\s*\n+/g, '\n');
    text = text.replace(/\s+([.,!?])/g, '$1');
    return text.trim();
}

async function paraphraseSentence(sentence) {
    if (!textGenerator) {
        console.warn('[CASI] Text generator not initialized, returning original sentence');
        return sentence;
    }
    try {
        const generated = await textGenerator(`Paraphrase: ${sentence}`, { max_new_tokens: 50, do_sample: true });
        return generated[0].generated_text.replace(/^Paraphrase: /, '').trim();
    } catch (error) {
        console.error('[CASI] Paraphrase error:', error.message);
        return sentence;
    }
}

function formatResponse(text, maxWords, mood) {
    text = preserveFormatting(text);
    
    text = text
        .replace(/\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s+\d{4}\b/g, '')
        .replace(/[^\w\s.,!?*'\n|-]/g, '')
        .trim();
    
    const doc = nlp(text);
    let sentences = doc.sentences().out('array');
    if (sentences.length === 0) sentences = ['Let’s explore this topic together.'];
    
    const lines = text.split('\n').filter(line => line.trim());
    let isList = lines.some(line => /^\s*(\d+\.\s+|[-*]\s+)/.test(line));
    let isTable = lines.some(line => /^\s*\|.*\|\s*$/.test(line));
    
    if (!isList && !isTable && mood === 'enthusiastic') {
        sentences = sentences.map(s => s.replace(/[.!?]$/, '!'));
    } else if (!isList && !isTable && mood === 'empathetic') {
        sentences = sentences.map(s => s.replace(/[.!?]$/, '.'));
    }
    
    let outputLines = [];
    let wordCount = 0;
    if (isList || isTable) {
        for (const line of lines) {
            const words = line.split(' ');
            if (wordCount + words.length <= maxWords) {
                outputLines.push(line);
                wordCount += words.length;
            } else {
                break;
            }
        }
    } else {
        for (const sentence of sentences) {
            const words = sentence.split(' ');
            if (wordCount + words.length <= maxWords) {
                outputLines.push(sentence);
                wordCount += words.length;
            } else {
                break;
            }
        }
    }
    
    let finalText = outputLines.join(isList || isTable ? '\n' : ' ');
    if (!finalText) finalText = 'Let’s dive into this topic!';
    
    if (!isList && !isTable) {
        finalText = finalText.charAt(0).toUpperCase() + finalText.slice(1);
        if (!/[.!?]/.test(finalText.slice(-1))) finalText += '.';
    }
    
    const finalDoc = nlp(finalText);
    finalText = finalDoc.text();
    
    return finalText;
}

function correctTypos(prompt) {
    const dictionary = new Typo('en_US');
    return prompt
        .split(' ')
        .map(word => {
            if (!dictionary.check(word)) {
                const suggestions = dictionary.suggest(word);
                return suggestions.length > 0 ? suggestions[0] : word;
            }
            return word;
        })
        .join(' ');
}

async function generateOriginalSentence(prompt, entities, keywords, templateType = 'explanation') {
    try {
        const template = handlebars.compile(templates[templateType]);
        const topic = entities[0] || keywords[0] || prompt.split(' ')[0];
        const context = keywords[1] || 'this topic';
        let data = {};
        
        const baseSentence = textGenerator
            ? (await textGenerator(`Explain ${topic} in one sentence`, { max_new_tokens: 50, do_sample: true }))[0].generated_text.trim()
            : `${topic} is a key concept in ${context}.`;
        
        switch (templateType) {
            case 'explanation':
                data = {
                    topic,
                    description: baseSentence,
                    examples: keywords.slice(2, 4).join(', ') || context
                };
                break;
            case 'definition':
                data = {
                    term: topic,
                    definition: baseSentence,
                    context: keywords[2] || context
                };
                break;
            case 'story':
                data = {
                    title: `The Story of ${topic}`,
                    intro: baseSentence,
                    body: `It involves ${context} and connects to ${keywords[2] || 'various ideas'}.`,
                    conclusion: `${topic} shapes our understanding of ${context}.`
                };
                break;
            case 'qa':
                data = {
                    question: prompt,
                    answer: baseSentence
                };
                break;
            case 'summary':
                data = {
                    topic,
                    overview: baseSentence,
                    points: keywords.slice(2, 5).map(k => `- ${k}`).join('\n') || `- ${context}`
                };
                break;
            case 'tutorial':
                data = {
                    topic,
                    objective: `Learn ${topic} effectively`,
                    steps: [
                        { index: 1, step: baseSentence },
                        { index: 2, step: `Apply ${context} in practice.` }
                    ],
                    tips: `Focus on ${keywords[2] || 'key concepts'}.`
                };
                break;
            case 'pros_cons':
                data = {
                    topic,
                    pros: [baseSentence, `Supports ${context}.`],
                    cons: [`Requires understanding ${keywords[2] || 'complexity'}.`],
                    verdict: `${topic} is valuable but needs careful application.`
                };
                break;
            case 'faq':
                data = {
                    topic,
                    questions: [
                        { index: 1, question: `What is ${topic}?`, answer: baseSentence },
                        { index: 2, question: `How does ${topic} work?`, answer: `It involves ${context}.` }
                    ]
                };
                break;
            case 'timeline':
                data = {
                    topic,
                    events: [
                        { year: 'Early', event: baseSentence },
                        { year: 'Recent', event: `Advances in ${context}.` }
                    ]
                };
                break;
            case 'code_snippet':
                data = {
                    language: keywords[2] || 'python',
                    topic,
                    code: `# Example for ${topic}\nprint("${baseSentence}")`,
                    explanation: `This code demonstrates ${context}.`
                };
                break;
            case 'case_study':
                data = {
                    topic,
                    background: baseSentence,
                    analysis: `It relates to ${context} through practical applications.`,
                    outcome: `${topic} achieved significant results.`
                };
                break;
            case 'recommendation':
                data = {
                    topic,
                    items: [
                        { index: 1, item: baseSentence },
                        { index: 2, item: `Leverage ${context} for better results.` }
                    ],
                    rationale: `Based on ${keywords[2] || 'analysis'}.`
                };
                break;
            case 'interview':
                data = {
                    topic,
                    questions: [
                        { index: 1, question: `What is ${topic}?`, answer: baseSentence },
                        { index: 2, question: `Why is ${topic} important?`, answer: `It drives ${context}.` }
                    ]
                };
                break;
            case 'glossary':
                data = {
                    topic,
                    terms: [
                        { term: topic, definition: baseSentence },
                        { term: context, definition: `Related to ${keywords[2] || 'concepts'}.` }
                    ]
                };
                break;
            case 'troubleshooting':
                data = {
                    topic,
                    issues: [
                        { index: 1, issue: `Understanding ${topic}`, solution: baseSentence },
                        { index: 2, issue: `Applying ${topic}`, solution: `Practice with ${context}.` }
                    ],
                    tips: `Focus on ${keywords[2] || 'practical steps'}.`
                };
                break;
            case 'roadmap':
                data = {
                    topic,
                    phases: [
                        { index: 1, phase: baseSentence, timeline: 'Short-term' },
                        { index: 2, phase: `Scale ${context}`, timeline: 'Long-term' }
                    ],
                    goals: `Achieve mastery in ${topic}.`
                };
                break;
            case 'analysis':
                data = {
                    topic,
                    overview: baseSentence,
                    findings: [`Related to ${context}.`, `Impacts ${keywords[2] || 'outcomes'}.`],
                    conclusion: `${topic} drives innovation.`
                };
                break;
            case 'fallback':
                data = { topic, context };
                break;
        }
        
        return template(data);
    } catch (error) {
        console.error('[CASI] Generate original sentence error:', error.message);
        return `Exploring ${topic}. Let's dive into ${context}!`;
    }
}

async function synthesizeFromOkeyAI(okeyText, prompt, patterns) {
    try {
        okeyText = preserveFormatting(okeyText);
        
        const doc = nlp(okeyText);
        const sentences = doc.sentences().out('array');
        const entities = doc.topics().out('array').concat(doc.people().out('array'), doc.places().out('array'));
        const promptDoc = nlp(prompt);
        const promptEntities = promptDoc.topics().out('array').concat(promptDoc.people().out('array'), promptDoc.places().out('array'));

        const allSentences = sentences.concat(patterns.map(p => p.text));
        const embeddings = await computeEmbeddings([prompt, ...allSentences]);
        const promptEmbedding = embeddings[0];
        const sentenceEmbeddings = embeddings.slice(1);

        const rankedSentences = allSentences.map((sentence, index) => {
            const similarity = lodash.sum(promptEmbedding.map((a, i) => a * sentenceEmbeddings[index][i])) /
                (Math.sqrt(lodash.sum(promptEmbedding.map(a => a * a))) * Math.sqrt(lodash.sum(sentenceEmbeddings[index].map(b => b * b))) || 1);
            return { sentence, score: similarity };
        }).sort((a, b) => b.score - a.score);

        let bestMatch = null;
        let bestScore = 0;
        for (const p of patterns) {
            const patternEmbedding = (await computeEmbeddings([p.text]))[0];
            const similarity = lodash.sum(promptEmbedding.map((a, i) => a * patternEmbedding[i])) /
                (Math.sqrt(lodash.sum(promptEmbedding.map(a => a * a))) * Math.sqrt(lodash.sum(patternEmbedding.map(b => b * b))) || 1);
            if (similarity > bestScore) {
                bestScore = similarity;
                bestMatch = p;
            }
        }

        const intents = [
            { type: 'list', keywords: ['list', 'types', 'examples'] },
            { type: 'comparison', keywords: ['compare', 'versus', 'vs'] },
            { type: 'definition', keywords: ['define', 'definition', 'what is'] },
            { type: 'story', keywords: ['story', 'tell me about', 'history'] },
            { type: 'qa', keywords: ['what is', 'how', 'why'] },
            { type: 'summary', keywords: ['summarize', 'summary', 'overview'] },
            { type: 'table', keywords: ['table', 'chart'] },
            { type: 'tutorial', keywords: ['tutorial', 'guide', 'how to'] },
            { type: 'pros_cons', keywords: ['pros and cons', 'advantages', 'disadvantages'] },
            { type: 'faq', keywords: ['faq', 'questions', 'answers'] },
            { type: 'timeline', keywords: ['timeline', 'history', 'chronology'] },
            { type: 'code_snippet', keywords: ['code', 'program', 'script'] },
            { type: 'case_study', keywords: ['case study', 'example', 'scenario'] },
            { type: 'recommendation', keywords: ['recommend', 'suggest', 'best practices'] },
            { type: 'interview', keywords: ['interview', 'q&a', 'discussion'] },
            { type: 'glossary', keywords: ['glossary', 'terms', 'definitions'] },
            { type: 'troubleshooting', keywords: ['troubleshoot', 'fix', 'issues'] },
            { type: 'roadmap', keywords: ['roadmap', 'plan', 'strategy'] },
            { type: 'analysis', keywords: ['analyze', 'analysis', 'evaluate'] }
        ];
        const intent = intents.find(i => i.keywords.some(k => prompt.toLowerCase().includes(k)))?.type || 'explanation';

        let base = '';
        if (intent === 'list' && rankedSentences.length > 0) {
            const listItems = rankedSentences.slice(0, 3).map((item, index) => ({
                index: index + 1,
                name: entities[index] || `Item ${index + 1}`,
                description: item.sentence
            }));
            const template = handlebars.compile(templates.list);
            base = template({ items: listItems });
            base += '\n' + await generateOriginalSentence(prompt, entities, [], 'summary');
        } else if (intent === 'comparison' && rankedSentences.length > 1) {
            const template = handlebars.compile(templates.comparison);
            base = template({
                item1: entities[0] || 'First concept',
                item2: entities[1] || 'Second concept',
                desc1: rankedSentences[0].sentence,
                desc2: rankedSentences[1].sentence,
                summary: await paraphraseSentence(rankedSentences.slice(0, 2).map(item => item.sentence).join(' '))
            });
            base += ' ' + await generateOriginalSentence(prompt, entities, [], 'explanation');
        } else if (intent === 'definition') {
            base = await generateOriginalSentence(prompt, entities, [], 'definition');
            if (bestMatch) {
                base += ' ' + await paraphraseSentence(bestMatch.text);
            }
        } else if (intent === 'story') {
            base = await generateOriginalSentence(prompt, entities, [], 'story');
            if (rankedSentences.length > 0) {
                base += '\n' + rankedSentences.slice(0, 2).map(item => item.sentence).join(' ');
            }
        } else if (intent === 'qa') {
            base = await generateOriginalSentence(prompt, entities, [], 'qa');
            if (bestMatch) {
                base += ' ' + await paraphraseSentence(bestMatch.text);
            }
        } else if (intent === 'summary') {
            base = await generateOriginalSentence(prompt, entities, [], 'summary');
            if (rankedSentences.length > 0) {
                base += '\n' + rankedSentences.slice(0, 2).map(item => item.sentence).join(' ');
            }
        } else if (intent === 'table' && rankedSentences.length > 0) {
            const rows = rankedSentences.slice(0, 3).map((item, index) => ({
                name: entities[index] || `Item ${index + 1}`,
                description: item.sentence
            }));
            const template = handlebars.compile(templates.table);
            base = template({ rows });
            base += '\n' + await generateOriginalSentence(prompt, entities, [], 'summary');
        } else if (intent === 'tutorial') {
            base = await generateOriginalSentence(prompt, entities, [], 'tutorial');
            if (bestMatch) {
                base += '\n' + await paraphraseSentence(bestMatch.text);
            }
        } else if (intent === 'pros_cons') {
            base = await generateOriginalSentence(prompt, entities, [], 'pros_cons');
            if (rankedSentences.length > 0) {
                base += '\n' + rankedSentences.slice(0, 2).map(item => item.sentence).join(' ');
            }
        } else if (intent === 'faq') {
            base = await generateOriginalSentence(prompt, entities, [], 'faq');
            if (bestMatch) {
                base += '\n' + await paraphraseSentence(bestMatch.text);
            }
        } else if (intent === 'timeline') {
            base = await generateOriginalSentence(prompt, entities, [], 'timeline');
            if (rankedSentences.length > 0) {
                base += '\n' + rankedSentences.slice(0, 2).map(item => item.sentence).join(' ');
            }
        } else if (intent === 'code_snippet') {
            base = await generateOriginalSentence(prompt, entities, [], 'code_snippet');
            if (bestMatch) {
                base += '\n' + await paraphraseSentence(bestMatch.text);
            }
        } else if (intent === 'case_study') {
            base = await generateOriginalSentence(prompt, entities, [], 'case_study');
            if (bestMatch) {
                base += '\n' + await paraphraseSentence(bestMatch.text);
            }
        } else if (intent === 'recommendation') {
            base = await generateOriginalSentence(prompt, entities, [], 'recommendation');
            if (bestMatch) {
                base += '\n' + await paraphraseSentence(bestMatch.text);
            }
        } else if (intent === 'interview') {
            base = await generateOriginalSentence(prompt, entities, [], 'interview');
            if (bestMatch) {
                base += '\n' + await paraphraseSentence(bestMatch.text);
            }
        } else if (intent === 'glossary') {
            base = await generateOriginalSentence(prompt, entities, [], 'glossary');
            if (bestMatch) {
                base += '\n' + await paraphraseSentence(bestMatch.text);
            }
        } else if (intent === 'troubleshooting') {
            base = await generateOriginalSentence(prompt, entities, [], 'troubleshooting');
            if (bestMatch) {
                base += '\n' + await paraphraseSentence(bestMatch.text);
            }
        } else if (intent === 'roadmap') {
            base = await generateOriginalSentence(prompt, entities, [], 'roadmap');
            if (bestMatch) {
                base += '\n' + await paraphraseSentence(bestMatch.text);
            }
        } else if (intent === 'analysis') {
            base = await generateOriginalSentence(prompt, entities, [], 'analysis');
            if (bestMatch) {
                base += '\n' + await paraphraseSentence(bestMatch.text);
            }
        } else {
            base = rankedSentences.slice(0, 10).map(item => item.sentence).join(' ');
            if (bestMatch) {
                base += ' ' + await paraphraseSentence(bestMatch.text);
            }
            base += ' ' + await generateOriginalSentence(prompt, entities, [], 'explanation');
        }

        let result = base;
        if (promptEntities.length > 0 && !result.toLowerCase().includes(promptEntities[0].toLowerCase())) {
            result = `${promptEntities[0]}: ${result}`;
        }

        result = result.replace(/\b(\w+)(?:\s+\1\b)+/gi, '$1');
        result = result.replace(/\s{2,}/g, ' ').trim();
        if (!result.split('\n').some(line => /^\s*(\d+\.\s+|[-*]\s+|\|.*\|)/.test(line))) {
            result = result.charAt(0).toUpperCase() + result.slice(1);
            if (!/[.!?]$/.test(result)) result += '.';
        }

        return result;
    } catch (error) {
        console.error('[CASI] Synthesize error:', error.message);
        return await generateOriginalSentence(prompt, [], [], 'fallback');
    }
}

async function fetchOkeyAIResponse(prompt) {
    const apiUrl = 'https://api.okeymeta.com.ng/api/ssailm/model/okeyai3.0-vanguard/okeyai';
    const params = { input: prompt };
    try {
        const response = await axios.get(apiUrl, { params, timeout: 15000 });
        let aiData = response.data;
        if (typeof aiData === 'string') {
            try {
                aiData = JSON.parse(aiData);
            } catch {
                return '';
            }
        }
        const contentRaw = aiData && (aiData.response || aiData?.model?.response);
        if (!contentRaw || typeof contentRaw !== 'string' || contentRaw.trim().length < 5) {
            return '';
        }
        return contentRaw.trim();
    } catch (error) {
        console.error('[CASI] OkeyAI fetch error:', error.message);
        return '';
    }
}

async function generateResponse({ prompt, diversityFactor = 0.7, depth = 8, breadth = 6, maxWords = 300, mood = 'neutral', patterns }) {
    try {
        if (!prompt || typeof prompt !== 'string' || prompt.trim().length < 1) {
            throw new Error('Prompt must be a non-empty string');
        }

        if (!nlpManager || !patternsCache.length) {
            await loadPatternsAndTrainModels();
        }

        const cleanedPrompt = preprocessPrompt(prompt);
        const okeyText = await fetchOkeyAIResponse(cleanedPrompt);
        let synthesized = '';
        if (okeyText) {
            synthesized = await synthesizeFromOkeyAI(okeyText, cleanedPrompt, patternsCache);
            await client.connect();
            const db = client.db('CASIDB');
            const patternsCol = db.collection('patterns');
            await patternsCol.insertOne({
                concept: cleanedPrompt,
                content: zlib.gzipSync(okeyText).toString('base64'),
                entities: [],
                sentiment: 0,
                updatedAt: new Date(),
                confidence: 0.97,
                source: 'OkeyMetaAI',
                title: `OkeyMetaAI: ${cleanedPrompt.slice(0, 50)}`
            });
            const paraphrased = await paraphraseSentence(okeyText);
            await patternsCol.insertOne({
                concept: cleanedPrompt,
                content: zlib.gzipSync(paraphrased).toString('base64'),
                entities: [],
                sentiment: 0,
                updatedAt: new Date(),
                confidence: 0.95,
                source: 'CASI_Paraphrased',
                title: `Paraphrased: ${cleanedPrompt.slice(0, 50)}`
            });
            learningEmitter.emit('newPattern', {
                concept: cleanedPrompt,
                content: zlib.gzipSync(okeyText).toString('base64')
            });
            learningEmitter.emit('newPattern', {
                concept: cleanedPrompt,
                content: zlib.gzipSync(paraphrased).toString('base64')
            });
            console.log('[CASI] Learned from OkeyAI:', okeyText);
            console.log('[CASI] Stored paraphrased:', paraphrased);
        } else {
            synthesized = await generateOriginalSentence(cleanedPrompt, [], [], 'fallback');
            if (patternsCache.length > 0) {
                const bestMatch = patternsCache[0];
                synthesized += ' ' + await paraphraseSentence(bestMatch.text);
            }
        }

        const nlpResult = await nlpManager.process('en', cleanedPrompt);

        let fallbackText = '';
        if (patterns && patterns.length > 0) {
            const best = patterns.find(p =>
                (nlpResult.intent && p.concept === nlpResult.intent) ||
                (p.concept && cleanedPrompt.toLowerCase().includes(p.concept.toLowerCase()))
            );
            if (best) {
                try {
                    fallbackText = zlib.gunzipSync(Buffer.from(best.content, 'base64')).toString();
                } catch {
                    fallbackText = '';
                }
            }
        }

        let outputText = synthesized || fallbackText || await generateOriginalSentence(cleanedPrompt, [], [], 'fallback');
        outputText = formatResponse(outputText, maxWords, mood);

        const confidence = okeyText ? 0.98 : 0.9;
        const outputId = crypto.randomBytes(12).toString('hex');

        await client.connect();
        const db = client.db('CASIDB');
        const patternsCol = db.collection('patterns');
        await patternsCol.insertOne({
            concept: cleanedPrompt,
            content: zlib.gzipSync(outputText).toString('base64'),
            entities: [],
            sentiment: 0,
            updatedAt: new Date(),
            confidence,
            source: 'CASI',
            title: `Response to: ${cleanedPrompt.slice(0, 50)}`
        });

        learningEmitter.emit('newPattern', {
            concept: cleanedPrompt,
            content: zlib.gzipSync(outputText).toString('base64')
        });

        return { text: outputText, confidence, outputId };
    } catch (error) {
        console.error('[CASI] Generate response error:', error.message);
        throw { message: `Generation failed: ${error.message}`, status: 500 };
    } finally {
        await client.close();
    }
}

module.exports = { generateResponse, loadPatternsAndTrainModels, initializeModels };