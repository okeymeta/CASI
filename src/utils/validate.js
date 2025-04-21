const lodash = require('lodash');

function validatePrompt(prompt) {
    if (!lodash.isString(prompt) || prompt.length < 1 || prompt.length > 1000) {
        console.error('Invalid prompt length:', prompt);
        throw { error: 'Invalid prompt: must be string, 1-1000 chars', status: 400, timestamp: new Date() };
    }
    return lodash.escape(prompt);
}

function validateGenerationParams({
    diversityFactor = 0.5,
    depth = 10,
    breadth = 5,
    maxWords = 100,
    mood = 'neutral'
}) {
    if (!Number.isFinite(diversityFactor) || diversityFactor < 0 || diversityFactor > 1) {
        console.error('Invalid diversityFactor:', diversityFactor);
        throw { error: 'Invalid diversityFactor: must be 0-1', status: 400, timestamp: new Date() };
    }
    if (!Number.isInteger(depth) || depth < 1 || depth > 20) {
        console.error('Invalid depth:', depth);
        throw { error: 'Invalid depth: must be integer, 1-20', status: 400, timestamp: new Date() };
    }
    if (!Number.isInteger(breadth) || breadth < 1 || breadth > 10) {
        console.error('Invalid breadth:', breadth);
        throw { error: 'Invalid breadth: must be integer, 1-10', status: 400, timestamp: new Date() };
    }
    if (!Number.isInteger(maxWords) || maxWords < 10 || maxWords > 500) {
        console.error('Invalid maxWords:', maxWords);
        throw { error: 'Invalid maxWords: must be 10-500', status: 400, timestamp: new Date() };
    }
    if (!['neutral', 'empathetic', 'playful'].includes(mood)) {
        console.error('Invalid mood:', mood);
        throw { error: 'Invalid mood: must be neutral, empathetic, or playful', status: 400, timestamp: new Date() };
    }
    return { diversityFactor, depth, breadth, maxWords, mood };
}

function validateUrl(url) {
    try {
        new URL(url);
        return lodash.escape(url);
    } catch (error) {
        console.error('Invalid URL:', url);
        throw { error: 'Invalid URL', status: 400, timestamp: new Date() };
    }
}

function validateFeedback({
    outputId,
    isAccurate,
    correctEmotion,
    quality,
    isVaried
}) {
    if (!lodash.isString(outputId) || outputId.length < 1) {
        console.error('Invalid outputId:', outputId);
        throw { error: 'Invalid outputId: must be non-empty string', status: 400, timestamp: new Date() };
    }
    if (!lodash.isBoolean(isAccurate)) {
        console.error('Invalid isAccurate:', isAccurate);
        throw { error: 'Invalid accuracy flag: must be boolean', status: 400, timestamp: new Date() };
    }
    if (correctEmotion && !lodash.isString(correctEmotion)) {
        console.error('Invalid correctEmotion:', correctEmotion);
        throw { error: 'Invalid emotion: must be string', status: 400, timestamp: new Date() };
    }
    if (!Number.isInteger(quality) || quality < 1 || quality > 5) {
        console.error('Invalid quality:', quality);
        throw { error: 'Invalid quality: must be 1-5', status: 400, timestamp: new Date() };
    }
    if (!lodash.isBoolean(isVaried)) {
        console.error('Invalid isVaried:', isVaried);
        throw { error: 'Invalid variation flag: must be boolean', status: 400, timestamp: new Date() };
    }
    return {
        outputId,
        isAccurate,
        correctEmotion: correctEmotion ? lodash.escape(correctEmotion) : undefined,
        quality,
        isVaried
    };
}

module.exports = {
    validatePrompt,
    validateGenerationParams,
    validateUrl,
    validateFeedback
};