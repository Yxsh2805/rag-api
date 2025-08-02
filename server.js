const express = require('express');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const app = express();

// CRITICAL FIX: Use PORT environment variable that Render provides
const port = process.env.PORT || 10000; // Changed from 3000 to 10000 to match your logs

// Middleware
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

// CORS middleware
app.use((req, res, next) => {
    res.header('Access-Control-Allow-Origin', '*');
    res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
    res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept, Authorization');
    if (req.method === 'OPTIONS') {
        res.sendStatus(200);
    } else {
        next();
    }
});

// Create Python script that will be executed
const createPythonScript = () => {
    const pythonCode = `
import os
import sys
import json
import warnings
from langchain_together import ChatTogether, TogetherEmbeddings
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableAssign, RunnableLambda
import chromadb
from langchain_chroma import Chroma
from concurrent.futures import ThreadPoolExecutor
import time
from datetime import datetime
import tempfile
import uuid

# Suppress all warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

def initialize_rag_engine():
    # Set environment variables
    os.environ["TOGETHER_API_KEY"] = "deb14836869b48e01e1853f49381b9eb7885e231ead3bc4f6bbb4a5fc4570b78"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = "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"
    os.environ["LANGCHAIN_PROJECT"] = "chunking and rag bajaj"

    # LLM model and embeddings
    chat_model = ChatTogether(model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free")
    embeddings = TogetherEmbeddings(model="BAAI/bge-base-en-v1.5")

    # Create temporary directory for this session
    temp_dir = tempfile.mkdtemp()
    vectorstore_path = os.path.join(temp_dir, f"vectorstore_{uuid.uuid4().hex[:8]}")
    
    # Chroma vector store with persistent client
    persistent_client = chromadb.PersistentClient(path=vectorstore_path)
    persistent_client.get_or_create_collection("my_collection")
    vectorstore = Chroma(
        client=persistent_client,
        collection_name="my_collection",
        embedding_function=embeddings,
    )
    retriever = vectorstore.as_retriever()

    # Prompt template
    policy_prompt = ChatPromptTemplate([
        ("system", '''You are a helpful assistant who is an expert in explaining insurance policies and their application to general queries.
Help the human with their queries related to the context of the policy provided. If you don't feel you have a concrete answer, say so.
Do not provide false information.

IMPORTANT OUTPUT FORMAT: You will receive multiple questions separated by " | ". Answer each question in the same order and separate your answers with " | ". Do not include any other separators, explanations, or formatting.

Example:
Input Questions: "What is the grace period for premium payment? | What is the waiting period for PED? | Does this policy cover maternity?"
Output Answers: "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits. | There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered. | Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months."

Guidelines:
- Keep answers short and direct, explain if needed—meat of the answer at the front.
- No generalizations; caveats must be clear and explicit.
- Example bad: "if adherence to certain rules and meets certain conditions" (too vague/generic).
- Example good: "if the insured is under the age of 60 and income is over 7lpa" (if supported by context!).
- Replicate definitions in the language of the source document.
- CRITICAL: Separate each answer with " | " and maintain the exact order of questions.'''),
        ("human", '''
Answer the following questions (separated by " | "):
{query}

Here are some relevant excerpts that might be useful for you in answering the questions:
{context}
'''),
    ])

    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=90)

    # Compose the RAG chain
    def build_chain(retriever, chat_model, policy_prompt):
        def retrieve(state):
            query = state["query"]
            results = retriever.invoke(query)
            return " ".join([doc.page_content for doc in results])
        return (
            RunnableAssign({"context": RunnableLambda(retrieve)}) |
            policy_prompt |
            chat_model |
            StrOutputParser()
        )
    rag_chain = build_chain(retriever, chat_model, policy_prompt)

    return {
        "text_splitter": text_splitter,
        "vectorstore": vectorstore,
        "retriever": retriever,
        "rag_chain": rag_chain,
        "persistent_client": persistent_client,
        "embeddings": embeddings
    }

def rag_answer_for_url(url, query, rag_engine, max_workers=8):
    try:
        loader = UnstructuredURLLoader(urls=[url])
        docs = loader.load()

        # Use smaller chunks for faster processing
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        chunks = text_splitter.split_documents(docs)
        
        # Limit chunks for speed
        chunks = chunks[:20]  # Only use first 20 chunks for speed
        
        vectorstore = rag_engine["vectorstore"]

        # Add chunks sequentially for reliability
        for chunk in chunks:
            vectorstore.add_documents([chunk])

        result = rag_engine["rag_chain"].invoke({"query": query})
        return result
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    try:
        # Read input from stdin
        input_data = sys.stdin.read()
        data = json.loads(input_data)
        
        document_url = data.get('documents', '')
        questions = data.get('questions', [])
        
        if not document_url or not questions:
            print(json.dumps({"error": "Missing documents URL or questions"}))
            return
        
        # Initialize RAG engine
        rag_engine = initialize_rag_engine()
        
        # Join questions with " | " separator as expected by the prompt
        combined_query = " | ".join(questions)
        
        # Get answer from RAG model
        result = rag_answer_for_url(document_url, combined_query, rag_engine)
        
        # Split the result back into individual answers
        answers = [answer.strip() for answer in result.split(" | ")]
        
        # Ensure we have the same number of answers as questions
        while len(answers) < len(questions):
            answers.append("Unable to find relevant information in the document.")
        
        # Return JSON response
        response = {"answers": answers[:len(questions)]}
        print(json.dumps(response))
        
    except Exception as e:
        print(json.dumps({"error": f"An error occurred: {str(e)}"}))

if __name__ == "__main__":
    main()
`;

    const scriptPath = path.join(__dirname, 'rag_script.py');
    fs.writeFileSync(scriptPath, pythonCode);
    return scriptPath;
};

// Initialize Python script
const pythonScriptPath = createPythonScript();

// Health check endpoint
app.get('/health', (req, res) => {
    res.json({ 
        status: 'healthy', 
        timestamp: new Date().toISOString(),
        service: 'RAG API',
        port: port,
        env: process.env.NODE_ENV || 'development'
    });
});

// Main RAG endpoint
app.post('/rag/query', async (req, res) => {
    try {
        const { documents, questions } = req.body;

        // Validate input
        if (!documents || !questions || !Array.isArray(questions) || questions.length === 0) {
            return res.status(400).json({
                error: 'Invalid input. Expected format: {"documents": "url", "questions": ["question1", "question2", ...]}'
            });
        }

        // Validate URL format
        try {
            new URL(documents);
        } catch (e) {
            return res.status(400).json({
                error: 'Invalid document URL provided'
            });
        }

        console.log(`Processing ${questions.length} questions for document: ${documents}`);

        // Prepare input for Python script
        const inputData = JSON.stringify({ documents, questions });

        // Execute Python script (try multiple Python commands)
        const pythonCommands = process.platform === 'win32' 
            ? ['python', 'py', 'python3'] 
            : ['python3', 'python', 'py'];
        
        let pythonProcess;
        let pythonCommand = pythonCommands[0]; // Default to first option
        
        // Try to find working Python command
        for (const cmd of pythonCommands) {
            try {
                pythonProcess = spawn(cmd, [pythonScriptPath]);
                pythonCommand = cmd;
                break;
            } catch (error) {
                console.log(`${cmd} not found, trying next...`);
                continue;
            }
        }
        
        if (!pythonProcess) {
            return res.status(500).json({
                error: 'Python interpreter not found',
                details: 'Please ensure Python is installed and accessible'
            });
        }
        
        let output = '';
        let errorOutput = '';
        
        pythonProcess.stdout.on('data', (data) => {
            output += data.toString();
        });
        
        pythonProcess.stderr.on('data', (data) => {
            errorOutput += data.toString();
        });
        
        pythonProcess.on('close', (code) => {
            if (code !== 0) {
                console.error('Python script error:', errorOutput);
                return res.status(500).json({
                    error: 'Internal server error while processing the request',
                    details: errorOutput
                });
            }
            
            try {
                // Clean the output by removing warnings and extracting JSON
                const lines = output.trim().split('\n');
                let jsonLine = '';
                
                // Find the line that looks like JSON (starts with { or [)
                for (const line of lines) {
                    if (line.trim().startsWith('{') || line.trim().startsWith('[')) {
                        jsonLine = line.trim();
                        break;
                    }
                }
                
                if (!jsonLine) {
                    throw new Error('No JSON found in output');
                }
                
                const result = JSON.parse(jsonLine);
                
                if (result.error) {
                    return res.status(500).json(result);
                }
                
                // Clean up the answers - remove trailing " |" if present
                if (result.answers) {
                    result.answers = result.answers.map(answer => 
                        answer.replace(/\s*\|\s*$/, '').trim()
                    );
                }
                
                console.log('Successfully processed request');
                res.json(result);
            } catch (parseError) {
                console.error('Error parsing Python output:', parseError);
                console.error('Raw output:', output);
                res.status(500).json({
                    error: 'Error parsing response from RAG model'
                });
            }
        });
        
        // Send input to Python script
        pythonProcess.stdin.write(inputData);
        pythonProcess.stdin.end();
        
        // Set timeout for long-running requests
        setTimeout(() => {
            pythonProcess.kill('SIGTERM');
            if (!res.headersSent) {
                res.status(408).json({
                    error: 'Request timeout. The document processing took too long.'
                });
            }
        }, 300000); // 5 minutes timeout
        
    } catch (error) {
        console.error('Server error:', error);
        res.status(500).json({
            error: 'Internal server error',
            message: error.message
        });
    }
});

// Error handling middleware
app.use((error, req, res, next) => {
    console.error('Unhandled error:', error);
    res.status(500).json({
        error: 'Internal server error',
        message: error.message
    });
});

// Handle 404
app.use((req, res) => {
    res.status(404).json({
        error: 'Endpoint not found',
        message: 'Please check the API documentation for available endpoints'
    });
});

// CRITICAL FIX: Proper server binding for Render
const server = app.listen(port, '0.0.0.0', () => {
    console.log(`RAG API server running on port ${port}`);
    console.log(`Health check: http://localhost:${port}/health`);
    console.log(`RAG endpoint: POST http://localhost:${port}/rag/query`);
    
    // Additional debug info for Render
    const address = server.address();
    console.log(`Server bound to ${address.address}:${address.port}`);
    console.log(`Environment: ${process.env.NODE_ENV || 'development'}`);
    console.log(`Process ID: ${process.pid}`);
});

// Handle server errors
server.on('error', (err) => {
    console.error('Server error:', err);
    if (err.code === 'EADDRINUSE') {
        console.error(`Port ${port} is already in use`);
    }
});

// Graceful shutdown
process.on('SIGTERM', () => {
    console.log('SIGTERM received, shutting down gracefully');
    server.close(() => {
        console.log('Process terminated');
    });
});

process.on('SIGINT', () => {
    console.log('SIGINT received, shutting down gracefully');
    server.close(() => {
        console.log('Process terminated');
    });
});

module.exports = app;
