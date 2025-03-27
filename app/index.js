import { pipeline } from '@xenova/transformers';
import { Hnswlib } from 'hnswlib-node';
import fetch from 'node-fetch';
import readline from 'readline';
import fs from 'fs/promises';
import path from 'path';

// Function to read jokes from markdown files
async function loadJokesFromDocs() {
    const docsDir = path.join(process.cwd(), 'docs');
    const files = await fs.readdir(docsDir);
    const jokes = [];

    for (const file of files) {
        if (file.endsWith('.md')) {
            const content = await fs.readFile(path.join(docsDir, file), 'utf-8');
            const lines = content.split('\n');
            const fileJokes = lines
                .filter(line => line.trim().startsWith('-'))
                .map(line => line.replace('-', '').trim());
            jokes.push(...fileJokes);
        }
    }

    return jokes;
}

// Initialize the embedding model
const embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');

// Load jokes from documentation
console.log('Loading jokes from documentation...');
const dadJokes = await loadJokesFromDocs();

// Initialize HNSWLib
const hnswlib = new Hnswlib();
const dimension = 384; // Dimension of the embeddings
const maxElements = dadJokes.length;

// Initialize the index
await hnswlib.initIndex(dimension, maxElements);

// Generate embeddings for all jokes
console.log('Generating embeddings for jokes...');
for (let i = 0; i < dadJokes.length; i++) {
    const embedding = await embedder(dadJokes[i], {
        pooling: 'mean',
        normalize: true
    });
    await hnswlib.addPoint(embedding.data, i);
}

// Function to query Ollama
async function queryOllama(prompt) {
    const response = await fetch('http://localhost:11434/api/generate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            model: 'mistral',
            prompt: prompt,
            stream: false
        })
    });
    const data = await response.json();
    return data.response;
}

// Function to get relevant jokes
async function getRelevantJokes(query, k = 3) {
    const queryEmbedding = await embedder(query, {
        pooling: 'mean',
        normalize: true
    });
    const results = await hnswlib.searchKnn(queryEmbedding.data, k);
    return results.neighbors.map(index => dadJokes[index]);
}

// Main chat loop
const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

console.log('Welcome to the DAD Developer Jokes Chatbot!');
console.log('Ask me about programming jokes or type "exit" to quit.');

async function chat() {
    rl.question('You: ', async (input) => {
        if (input.toLowerCase() === 'exit') {
            rl.close();
            return;
        }

        try {
            // Get relevant jokes
            const relevantJokes = await getRelevantJokes(input);
            
            // Create context with relevant jokes
            const context = `Here are some relevant programming dad jokes:\n${relevantJokes.join('\n')}\n\nUser question: ${input}\n\nPlease provide a response that incorporates these jokes or creates a new programming dad joke based on the user's question.`;

            // Get response from Ollama
            const response = await queryOllama(context);
            console.log('\nBot:', response);
            
            // Continue the chat
            chat();
        } catch (error) {
            console.error('Error:', error.message);
            chat();
        }
    });
}

chat(); 