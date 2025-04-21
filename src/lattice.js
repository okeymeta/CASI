const { MongoClient } = require('mongodb');

const mongoUri = 'mongodb+srv://nwaozor:nwaozor@cluster0.rmvi7qm.mongodb.net/CASIDB?retryWrites=true&w=majority';
const client = new MongoClient(mongoUri);

async function init() {
    try {
        await client.connect();
        const db = client.db('CASIDB');
        const patterns = db.collection('patterns');

        // Drop concept_1 index if it exists
        try {
            await patterns.dropIndex('concept_1');
            console.log('Dropped concept_1 index');
        } catch (error) {
            if (error.codeName === 'IndexNotFound') {
                console.log('concept_1 index not found, proceeding');
            } else {
                console.warn('Failed to drop concept_1 index:', error.message);
            }
        }

        // Create non-unique index on concept for performance
        await patterns.createIndex({ concept: 1 }, { unique: false });
        console.log('Created non-unique index on concept');

        return { patterns };
    } catch (error) {
        console.error('Lattice initialization failed:', error.message);
        throw new Error('Lattice initialization failed');
    }
}

module.exports = { init };