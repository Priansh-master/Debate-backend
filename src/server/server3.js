import express from 'express';
import mongoose from 'mongoose';
import cors from 'cors';
import dotenv from 'dotenv';
import { ChatGroq } from "@langchain/groq";
import { ChatPromptTemplate } from "@langchain/core/prompts";

// --- RAG IMPORTS ---
import { StringOutputParser } from "@langchain/core/output_parsers";
import { formatDocumentsAsString } from "langchain/util/document";
import { RunnableSequence } from "@langchain/core/runnables";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/hf_transformers";
import { FaissStore } from "@langchain/community/vectorstores/faiss";

// Load environment variables from .env file
dotenv.config();

const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(express.json());

const MONGODB_URI = process.env.MONGO_URI;

if (!MONGODB_URI) {
  console.error('FATAL ERROR: MONGO_URI is not defined in the .env file.');
  process.exit(1);
}

// MongoDB Atlas Connection
const connectDB = async () => {
  try {
    await mongoose.connect(MONGODB_URI);
    console.log('Successfully connected to MongoDB Atlas');
  } catch (error) {
    console.error('MongoDB connection error:', error);
    process.exit(1);
  }
};

connectDB();

// Debate Schema - No changes needed here.
const debateSchema = new mongoose.Schema({
  clientId: { type: String, required: true, index: true },
  debateTopic: { type: String, required: true },
  userRole: { type: String, required: true },
  chatHistory: [{
    speaker: String,
    content: String,
    timestamp: Date
  }],
  adjudicationResult: { type: Object, default: {} },
  uploadedFiles: [{
    filename: String,
    data: Buffer,
    mimetype: String,
  }],
  createdAt: { type: Date, default: Date.now }
});

const Debate = mongoose.model('Debate', debateSchema);

// --- Routes (No changes to these routes) ---

app.get('/api/debates', async (req, res) => {
  try {
    const debates = await Debate.find(
      {},
      'debateTopic userRole createdAt clientId'
    ).sort({ createdAt: -1 });
    res.json({ success: true, data: debates });
  } catch (error) {
    console.error('Error fetching all debates:', error);
    res.status(500).json({
      success: false,
      message: 'An error occurred while fetching debates.',
      error: error.message
    });
  }
});

app.get('/api/debates/:debateId', async (req, res) => {
  try {
    const { debateId } = req.params;
    if (!mongoose.Types.ObjectId.isValid(debateId)) {
      return res.status(400).json({ success: false, message: 'Invalid debate ID format.' });
    }
    const debate = await Debate.findById(debateId);
    if (!debate) {
      return res.status(404).json({ success: false, message: 'Debate not found.' });
    }
    res.json({ success: true, data: debate });
  } catch (error)
  {
    console.error('Error fetching debate details:', error);
    res.status(500).json({
      success: false,
      message: 'An error occurred while fetching debate details.',
      error: error.message
    });
  }
});


app.post('/api/debates', async (req, res) => {
  try {
    const newDebate = new Debate(req.body);
    const savedDebate = await newDebate.save();
    res.status(201).json({ success: true, data: savedDebate });
  } catch (error) {
    console.error('Error creating debate:', error);
    if (error.name === 'ValidationError') {
      return res.status(400).json({ success: false, message: 'Validation Error', error: error.message });
    }
    res.status(500).json({
      success: false,
      message: 'An error occurred while creating the debate.',
      error: error.message
    });
  }
});


// --- MODIFIED RAG Chat Endpoint ---
app.post('/api/chat/rag', async (req, res) => {
  const { question, clientId } = req.body;

  if (!question) {
    return res.status(400).json({ success: false, message: 'Question is required.' });
  }

  try {
    // === 1. LOAD ===
    const query = clientId ? { clientId } : {};
    const debates = await Debate.find(query).sort({ createdAt: -1 });

    if (debates.length === 0) {
      return res.json({
        success: true,
        reply: "I couldn't find any debate history. Once you complete a debate, you can ask me questions about it."
      });
    }

    const debateTexts = debates.map(debate => {
      const chatHistoryText = debate.chatHistory
        .map(chat => `${chat.speaker}: ${chat.content}`)
        .join('\n');
      return `Debate on "${debate.debateTopic}" (Client: ${debate.clientId}):\n${chatHistoryText}`;
    });

    // === 2. SPLIT ===
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });
    const splitDocs = await textSplitter.createDocuments(debateTexts);

    // === 3. EMBED & STORE ===
    const embeddings = new HuggingFaceTransformersEmbeddings({
      modelName: "Xenova/all-MiniLM-L6-v2",
    });
    const vectorStore = await FaissStore.fromDocuments(splitDocs, embeddings);

    // === 4. RETRIEVE ===
    const retriever = vectorStore.asRetriever({ k: 5 });

    // === 5. GENERATE ===
    const prompt = ChatPromptTemplate.fromMessages([
      ["system", "You are an expert assistant who analyzes a user's debate history. Answer the user's question based ONLY on the context provided below. If the information is not in the context, explicitly state that you cannot answer based on their history. Be concise and helpful.\n\nCONTEXT:\n{context}"],
      ["human", "{question}"],
    ]);

    const model = new ChatGroq({
      apiKey: process.env.GROQ_API_KEY,
      model: "openai/gpt-oss-20b",
    });
    
    // --- THIS IS THE CORRECTED PART ---
    const ragChain = RunnableSequence.from([
      {
        // The 'context' is now a sub-chain that first extracts the question string
        // before passing it to the retriever.
        context: ((input) => input.question).pipe(retriever).pipe(formatDocumentsAsString),
        // The 'question' is passed through unchanged.
        question: (input) => input.question,
      },
      prompt,
      model,
      new StringOutputParser(),
    ]);
    // --- END OF CORRECTION ---

    // Invoke the chain with the user's question.
    const result = await ragChain.invoke({ question: question });
    
    res.json({ success: true, reply: result });

  } catch (error) {
    console.error('RAG chat error:', error);
    res.status(500).json({
      success: false,
      message: 'An error occurred while processing your request.'
    });
  }
});

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({
    success: true,
    message: 'Server is running',
    timestamp: new Date().toISOString()
  });
});

app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
