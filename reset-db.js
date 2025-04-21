const { MongoClient } = require('mongodb');
const zlib = require('zlib');

const mongoUri = 'mongodb+srv://nwaozor:nwaozor@cluster0.rmvi7qm.mongodb.net/CASIDB?retryWrites=true&w=majority';
const client = new MongoClient(mongoUri);

async function dropIndexWithRetries(collection, indexName, maxRetries = 3) {
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
        try {
            await collection.dropIndex(indexName);
            console.log(`Successfully dropped ${indexName} index on attempt ${attempt}`);
            return true;
        } catch (error) {
            if (error.codeName === 'IndexNotFound') {
                console.log(`${indexName} index not found, proceeding`);
                return true;
            }
            console.warn(`Failed to drop ${indexName} index on attempt ${attempt}: ${error.message}`);
            if (attempt === maxRetries) throw error;
            await new Promise(resolve => setTimeout(resolve, 1000));
        }
    }
}

async function resetDatabase() {
    try {
        await client.connect();
        const db = client.db('CASIDB');
        const patterns = db.collection('patterns');

        // Drop concept_1 index with retries
        await dropIndexWithRetries(patterns, 'concept_1');

        // Verify no unique index on concept
        const indexes = await patterns.indexes();
        if (indexes.some(index => index.key.concept && index.unique)) {
            throw new Error('Unique index on concept field still exists after dropping');
        }
        console.log('Verified: No unique index on concept field');

        // Clear existing data
        await patterns.deleteMany({});
        console.log('Cleared patterns collection');

        // Insert diverse seed patterns
        const seedPatterns = [
            {
                concept: 'greeting',
                content: zlib.gzipSync('Hello! I’m CASI, here to assist with knowledge or a friendly chat.').toString('base64'),
                entities: [{ value: 'hello', type: 'greeting' }],
                sentiment: 0.5,
                updatedAt: new Date(),
                confidence: 0.95,
                source: 'seed',
                title: 'Greeting 1'
            },
            {
                concept: 'greeting',
                content: zlib.gzipSync('Hi! Ready to explore any topic or question?').toString('base64'),
                entities: [{ value: 'hi', type: 'greeting' }],
                sentiment: 0.5,
                updatedAt: new Date(),
                confidence: 0.95,
                source: 'seed',
                title: 'Greeting 2'
            },
            {
                concept: 'nervousness',
                content: zlib.gzipSync('Feeling nervous is normal before a presentation. Try deep breathing or rehearsing your speech.').toString('base64'),
                entities: [{ value: 'nervous', type: 'emotion' }, { value: 'presentation', type: 'event' }],
                sentiment: 0.2,
                updatedAt: new Date(),
                confidence: 0.95,
                source: 'seed',
                title: 'Nervousness Presentation Advice'
            },
            {
                concept: 'nervousness',
                content: zlib.gzipSync('Nervousness can be managed with practice and visualization techniques.').toString('base64'),
                entities: [{ value: 'nervous', type: 'emotion' }],
                sentiment: 0.2,
                updatedAt: new Date(),
                confidence: 0.95,
                source: 'seed',
                title: 'Nervousness Management'
            },
            {
                concept: 'artificial intelligence',
                content: zlib.gzipSync('Artificial intelligence enables machines to learn and solve problems like humans.').toString('base64'),
                entities: [{ value: 'artificial intelligence', type: 'concept' }],
                sentiment: 0.3,
                updatedAt: new Date(),
                confidence: 0.95,
                source: 'seed',
                title: 'AI Overview'
            },
            {
                concept: 'yoruba culture',
                content: zlib.gzipSync('Yoruba culture is vibrant, with festivals like Osun-Osogbo and rich traditions.').toString('base64'),
                entities: [{ value: 'yoruba culture', type: 'culture' }],
                sentiment: 0.4,
                updatedAt: new Date(),
                confidence: 0.95,
                source: 'seed',
                title: 'Yoruba Culture Overview'
            },
            {
                concept: 'nigerian tech',
                content: zlib.gzipSync('Nigerian tech is thriving with startups like Flutterwave and Paystack.').toString('base64'),
                entities: [{ value: 'nigerian tech', type: 'industry' }],
                sentiment: 0.4,
                updatedAt: new Date(),
                confidence: 0.95,
                source: 'seed',
                title: 'Nigerian Tech Overview'
            },
            {
                concept: 'mental health',
                content: zlib.gzipSync('Mental health is key; mindfulness and support can reduce stress.').toString('base64'),
                entities: [{ value: 'mental health', type: 'health' }],
                sentiment: 0.3,
                updatedAt: new Date(),
                confidence: 0.95,
                source: 'seed',
                title: 'Mental Health Advice'
            },
            {
                concept: 'education technology',
                content: zlib.gzipSync('Education technology enhances learning with tools like online platforms.').toString('base64'),
                entities: [{ value: 'education technology', type: 'technology' }],
                sentiment: 0.3,
                updatedAt: new Date(),
                confidence: 0.95,
                source: 'seed',
                title: 'EdTech Overview'
            },
            {
                concept: 'blockchain',
                content: zlib.gzipSync('Blockchain ensures secure, transparent transactions in tech.').toString('base64'),
                entities: [{ value: 'blockchain', type: 'technology' }],
                sentiment: 0.3,
                updatedAt: new Date(),
                confidence: 0.95,
                source: 'seed',
                title: 'Blockchain Overview'
            }
        ];

        await patterns.insertMany(seedPatterns);
        console.log(`Inserted ${seedPatterns.length} seed patterns`);

        // Verify indexes
        console.log('Current indexes:', indexes);
    } catch (error) {
        console.error('Failed to reset database:', error.message);
        process.exit(1);
    } finally {
        await client.close();
    }
}

resetDatabase();