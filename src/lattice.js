const graphlib = require('graphlib');
const { MongoClient } = require('mongodb');
const axios = require('axios');
const lodash = require('lodash');
const zlib = require('zlib');

const mongoUri = 'mongodb+srv://nwaozor:nwaozor@cluster0.rmvi7qm.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0';
const client = new MongoClient(mongoUri);
const graph = new graphlib.Graph({ directed: true, compound: true });

async function init() {
    try {
        await client.connect();
        const db = client.db('CASIDB');
        const patterns = db.collection('patterns');

        // Ensure indexes
        await patterns.createIndex({ concept: 1 }, { unique: true });
        await patterns.createIndex({ updatedAt: 1 }, { expireAfterSeconds: 2592000 });

        // Initialize graph with existing patterns
        const existingPatterns = await patterns.find({}).toArray();
        existingPatterns.forEach(pattern => {
            if (pattern.concept) {
                graph.setNode(pattern.concept, {
                    confidence: pattern.confidence || 0.9,
                    entities: pattern.entities || [],
                    sentiment: pattern.sentiment || 0
                });
            }
        });

        console.log('Lattice graph initialized with', graph.nodeCount(), 'nodes');
    } catch (error) {
        console.error('Failed to initialize lattice:', error);
        throw new Error('Lattice initialization failed');
    }
}

async function updateNode({ concept, updates }) {
    try {
        await client.connect();
        const db = client.db('CASIDB');
        const patterns = db.collection('patterns');

        const result = await patterns.updateOne(
            { concept },
            { $set: { ...updates, updatedAt: new Date() } },
            { upsert: true }
        );

        graph.setNode(concept, {
            confidence: updates.confidence || 0.9,
            verified: updates.verified || false
        });

        return result.modifiedCount || result.upsertedCount;
    } catch (error) {
        console.error('Failed to update node:', error);
        throw new Error('Node update failed');
    }
}

async function getRelatedConcepts(prompt) {
    try {
        const response = await axios.get(`https://www.google.com/search?q=${encodeURIComponent(prompt)}`);
        const concepts = lodash.uniq(
            response.data.match(/[A-Z][a-z]+ [A-Z][a-z]+/g) || []
        ).slice(0, 5);

        return concepts;
    } catch (error) {
        console.error('Failed to fetch related concepts:', error);
        return [];
    }
}

module.exports = {
    init,
    updateNode,
    getRelatedConcepts,
    graph
};